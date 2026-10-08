import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# ── helpers ───────────────────────────────────────────────────────────────────

def softmax(logits):
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def _grid(n=100):
    gx, gy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    return gx.ravel(), gy.ravel()


# ── Linear (Adam softmax regression) ─────────────────────────────────────────

class LinearModel:
    """Logistic regression with Adam optimiser.

    Converges fast and cleanly on linearly-separable data; its hard limit
    (a single hyperplane per pair of classes) is visually educational.
    """
    def __init__(self, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps

    def init(self, num_classes):
        self.nc = num_classes
        self.W = np.zeros((2, num_classes))
        self.b = np.zeros((1, num_classes))
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
        self.t = 0

    def train_epoch(self, X, y):
        self.t += 1
        N = len(y)
        logits = X @ self.W + self.b
        probs  = softmax(logits)
        loss   = -np.mean(np.log(probs[np.arange(N), y] + 1e-10))
        acc    = np.mean(np.argmax(logits, 1) == y)

        dz = probs.copy(); dz[np.arange(N), y] -= 1; dz /= N
        gW = X.T @ dz
        gb = dz.sum(0, keepdims=True)

        # Adam
        for p, m, v, g in [
            (self.W, self.mW, self.vW, gW),
            (self.b, self.mb, self.vb, gb),
        ]:
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * g**2
            mc = m / (1 - self.b1**self.t)
            vc = v / (1 - self.b2**self.t)
            p -= self.lr * mc / (np.sqrt(vc) + self.eps)

        return float(loss), float(acc)

    def predict_grid(self, gx, gy):
        X = np.column_stack([gx, gy])
        logits = X @ self.W + self.b
        return np.argmax(logits, 1), softmax(logits)


# ── Genetic Algorithm (multi-prototype, vectorised) ──────────────────────────

class GeneticClassifier:
    """Genetic algorithm with multiple prototypes per class.

    Each individual carries P prototypes per class. A query point is
    classified to the class of its nearest prototype (Voronoi). With P>=3
    the algorithm can solve XOR and other nonlinear problems.

    Tournament selection, adaptive mutation, random immigrants.
    """
    P = 8  # prototypes per class

    def __init__(self, pop=80, mutation_rate=0.25):
        self.pop_size = pop
        self.mr = mutation_rate

    def init(self, num_classes):
        self.nc  = num_classes
        self.gen = 0
        # population: (pop, nc, P, 2)  — 2D coordinates only
        self.population = np.random.rand(self.pop_size, num_classes, self.P, 2)
        self.fitness = np.zeros(self.pop_size)

    # vectorised fitness over the entire population in one call
    def _evaluate_all(self, X, y):
        N = len(X)
        # X: (N,2), pop: (pop, nc, P, 2)
        # broadcast -> (pop, nc, P, N, 2) distance, pick min over P
        pop = self.population                                    # (pop,nc,P,2)
        Xb  = X[None, None, None, :, :]                         # (1,1,1,N,2)
        pb  = pop[:, :, :, None, :]                             # (pop,nc,P,1,2)
        d2  = ((pb - Xb)**2).sum(-1)                            # (pop,nc,P,N)
        min_d = d2.min(axis=2)                                   # (pop,nc,N)
        preds = np.argmin(min_d, axis=1)                        # (pop,N)
        self.fitness = (preds == y[None, :]).mean(axis=1)       # (pop,)

    def train_epoch(self, X, y):
        self._evaluate_all(X, y)
        order = np.argsort(self.fitness)[::-1]
        self.population = self.population[order]
        self.fitness    = self.fitness[order]

        best = float(self.fitness[0])
        avg  = float(self.fitness.mean())
        strength = max(0.02, 0.25 * (1 - avg))   # adaptive mutation

        # tournament selection (vectorised)
        def tournament(k=4):
            idx = np.random.randint(0, self.pop_size, size=(self.pop_size, k))
            winners = idx[np.arange(self.pop_size), np.argmax(self.fitness[idx], axis=1)]
            return winners

        p1_idx = tournament(); p2_idx = tournament()
        p1 = self.population[p1_idx]; p2 = self.population[p2_idx]

        # uniform crossover per prototype slot
        mask = np.random.rand(*p1.shape) > 0.5
        children = np.where(mask, p1, p2)

        # mutation
        mut_mask = np.random.rand(*children.shape[:-1]) < self.mr
        noise    = np.random.randn(*children.shape) * strength
        children += noise * mut_mask[..., None]
        children  = np.clip(children, 0, 1)

        # elitism: keep top 2, fill rest with children, inject 6 random
        n_random = 6
        n_child  = self.pop_size - 2 - n_random
        new_pop  = np.empty_like(self.population)
        new_pop[:2]                         = self.population[:2]
        new_pop[2:2 + n_child]              = children[:n_child]
        new_pop[2 + n_child:]               = np.random.rand(n_random, self.nc, self.P, 2)
        self.population = new_pop

        self.gen += 1
        return float(1 - best), float(best)

    def predict_grid(self, gx, gy):
        best = self.population[0]                            # (nc,P,2)
        X    = np.column_stack([gx, gy])                    # (M,2)
        pb   = best[:, :, None, :]                          # (nc,P,1,2)
        Xb   = X[None, None, :, :]                          # (1,1,M,2)
        d2   = ((pb - Xb)**2).sum(-1)                       # (nc,P,M)
        min_d = d2.min(axis=1)                              # (nc,M)
        preds = np.argmin(min_d, axis=0)                    # (M,)
        # soft confidence: inverse-distance weighting
        inv   = 1.0 / (min_d + 1e-6)                       # (nc,M)
        probs = (inv / inv.sum(0, keepdims=True)).T         # (M,nc)
        return preds, probs


# ── KNN ──────────────────────────────────────────────────────────────────────

class KNNModel:
    def __init__(self, k=5):
        self.k = k
        self.model = None

    def init(self, num_classes):
        self.nc = num_classes
        self.model = None

    def train_epoch(self, X, y):
        n_classes = len(np.unique(y))
        pts_per_class = max(1, len(X) // n_classes)
        k = max(1, min(self.k, pts_per_class // 2 + 1))
        if k % 2 == 0: k = max(1, k - 1)   # keep odd to avoid ties
        self.model = KNeighborsClassifier(n_neighbors=k, weights='distance')
        self.model.fit(X, y)
        acc  = float(np.mean(self.model.predict(X) == y))
        return float(1 - acc), acc

    def predict_grid(self, gx, gy):
        X = np.column_stack([gx, gy])
        preds = self.model.predict(X)
        probs_raw = self.model.predict_proba(X)
        full = np.zeros((len(preds), self.nc))
        for i, c in enumerate(self.model.classes_):
            full[:, c] = probs_raw[:, i]
        return preds, full


# ── SVM ──────────────────────────────────────────────────────────────────────

class SVMModel:
    ANIMATION_STEPS = 30

    def __init__(self, kernel='rbf', C=1.0, gamma='scale', degree=3):
        if kernel not in {'linear', 'poly', 'rbf', 'sigmoid'}:
            raise ValueError('Kernel inválido')
        self.kernel = kernel
        self.C = max(0.001, float(C))
        self.gamma = gamma if gamma in {'scale', 'auto'} else max(0.0001, float(gamma))
        self.degree = max(1, min(8, int(degree)))
        self.epochs = 1 if kernel == 'linear' or (kernel == 'poly' and self.degree == 1) else self.ANIMATION_STEPS
        self.epoch = 0
        self.model = None

    def init(self, num_classes):
        self.nc    = num_classes
        self.epoch = 0
        self.model = None

    def _target_gamma(self, X):
        if self.gamma == 'auto':
            return 1 / X.shape[1]
        if self.gamma == 'scale':
            variance = X.var()
            return 1 / (X.shape[1] * variance) if variance else 1.0
        return self.gamma

    def train_epoch(self, X, y):
        if self.epoch == 0 or self.kernel == 'linear':
            kernel = 'linear'
            gamma = 'scale'
            degree = 1
        else:
            progress = self.epoch / (self.epochs - 1)
            target_gamma = self._target_gamma(X)
            gamma = target_gamma / (50 ** (1 - progress))
            degree = max(1, round(1 + (self.degree - 1) * progress))
            kernel = self.kernel

        self.model = SVC(kernel=kernel, C=self.C, gamma=gamma,
                         degree=degree,
                         decision_function_shape='ovr')
        self.model.fit(X, y)
        acc  = float(np.mean(self.model.predict(X) == y))
        loss = float(1 - acc)
        self.epoch += 1
        return loss, acc

    def predict_grid(self, gx, gy):
        X     = np.column_stack([gx, gy])
        preds = self.model.predict(X)
        df    = self.model.decision_function(X)
        if df.ndim == 1:
            df = np.column_stack([-df, df])
        full = np.zeros((len(preds), self.nc))
        for i, c in enumerate(self.model.classes_):
            if c < self.nc:
                full[:, c] = df[:, i]
        return preds, softmax(full)


# ── registry ─────────────────────────────────────────────────────────────────

MODELS = {
    'linear':  LinearModel,
    'svm':     SVMModel,
    'genetic': GeneticClassifier,
    'knn':     KNNModel,
}
