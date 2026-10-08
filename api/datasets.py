"""Example datasets for the decision boundary demo.

All datasets return points normalised to [0.1, 0.9] so they sit nicely
inside the canvas. Each point is { x, y, color } with color in 0..n_classes-1.
"""
import numpy as np
from sklearn.datasets import load_iris


def _norm(X: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0.12, 0.88] per feature."""
    lo, hi = X.min(0), X.max(0)
    span = np.where(hi - lo == 0, 1, hi - lo)
    return 0.12 + (X - lo) / span * 0.76


def _pack(X: np.ndarray, y: np.ndarray) -> list[dict]:
    return [{"x": float(X[i, 0]), "y": float(X[i, 1]), "color": int(y[i])}
            for i in range(len(y))]


# ── Gaussian blobs (2 classes) ────────────────────────────────────────────────

def gaussian(n=80, seed=1):
    rng = np.random.default_rng(seed)
    centers = [(0.28, 0.35), (0.72, 0.65)]
    Xs, ys = [], []
    for c, (cx, cy) in enumerate(centers):
        pts = rng.normal([cx, cy], 0.12, size=(n // 2, 2))
        Xs.append(pts); ys.append(np.full(n // 2, c))
    X = np.clip(np.vstack(Xs), 0, 1)
    return _pack(X, np.concatenate(ys))


# ── XOR (2 classes) ──────────────────────────────────────────────────────────

def xor(n=80, seed=2):
    rng = np.random.default_rng(seed)
    per = n // 4
    corners = [(0.25, 0.75, 0), (0.75, 0.25, 0),
               (0.25, 0.25, 1), (0.75, 0.75, 1)]
    Xs, ys = [], []
    for cx, cy, c in corners:
        pts = rng.normal([cx, cy], 0.09, size=(per, 2))
        Xs.append(pts); ys.append(np.full(per, c))
    X = np.clip(np.vstack(Xs), 0, 1)
    return _pack(X, np.concatenate(ys))


# ── Concentric circles (2 classes) ───────────────────────────────────────────

def circles(n=80, seed=3):
    rng = np.random.default_rng(seed)
    pts, labels = [], []
    for c, (r_lo, r_hi) in enumerate([(0.0, 0.22), (0.32, 0.46)]):
        for _ in range(n // 2):
            angle = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(r_lo, r_hi)
            pts.append([0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)])
            labels.append(c)
    X = np.clip(np.array(pts), 0.02, 0.98)
    return _pack(X, np.array(labels))


# ── Spiral (2 classes) ────────────────────────────────────────────────────────

def spiral(n=96, seed=4):
    rng = np.random.default_rng(seed)
    per = n // 2
    pts, labels = [], []
    for c in range(2):
        t = np.linspace(0.3, 3.5 * np.pi, per)
        r = t / (3.5 * np.pi)
        offset = np.pi * c
        x = 0.5 + r * np.cos(t + offset)
        y = 0.5 + r * np.sin(t + offset)
        noise = rng.normal(0, 0.025, size=(per, 2))
        arm = np.column_stack([x, y]) + noise
        pts.append(arm)
        labels.append(np.full(per, c))
    X = np.clip(np.vstack(pts), 0.02, 0.98)
    return _pack(X, np.concatenate(labels))


# ── Four Gaussian clusters (4 classes) ───────────────────────────────────────

def four_gaussians(n=80, seed=5):
    rng = np.random.default_rng(seed)
    centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    per = n // 4
    Xs, ys = [], []
    for c, (cx, cy) in enumerate(centers):
        pts = rng.normal([cx, cy], 0.09, size=(per, 2))
        Xs.append(pts); ys.append(np.full(per, c))
    X = np.clip(np.vstack(Xs), 0, 1)
    return _pack(X, np.concatenate(ys))


# ── Iris (3 classes, petal length x petal width) ──────────────────────────────

def iris():
    data = load_iris()
    # features 2 & 3: petal length and petal width — best separation
    X_raw = data.data[:, 2:4].astype(float)
    y = data.target.astype(int)
    X = _norm(X_raw)
    # swap axes so petal length goes left-right, width goes bottom-top
    X = np.column_stack([X[:, 0], 1 - X[:, 1]])
    return _pack(X, y)


# ── Brazil capitals by region (5 classes) ─────────────────────────────────────

def brasil():
    """27 state capitals plotted by (longitude, latitude), coloured by
    the 5 official regions. Real geographic data — tests whether models
    can recover region boundaries from spatial coordinates."""
    # (lat, lon, region)  region: 0=Norte 1=Nordeste 2=Centro-Oeste 3=Sudeste 4=Sul
    caps = [
        # Norte
        (-3.1,  -60.0, 0),   # Manaus
        (-1.5,  -48.5, 0),   # Belem
        (-8.8,  -63.9, 0),   # Porto Velho
        (-10.0, -67.8, 0),   # Rio Branco
        ( 0.0,  -51.1, 0),   # Macapa
        (-10.2, -48.3, 0),   # Palmas
        ( 2.8,  -60.7, 0),   # Boa Vista
        # Nordeste
        (-13.0, -38.5, 1),   # Salvador
        (-8.1,  -34.9, 1),   # Recife
        (-3.7,  -38.5, 1),   # Fortaleza
        (-2.5,  -44.3, 1),   # Sao Luis
        (-7.1,  -34.9, 1),   # Joao Pessoa
        (-5.8,  -35.2, 1),   # Natal
        (-9.7,  -35.7, 1),   # Maceio
        (-10.9, -37.1, 1),   # Aracaju
        (-5.1,  -42.8, 1),   # Teresina
        # Centro-Oeste
        (-15.8, -47.9, 2),   # Brasilia
        (-16.7, -49.3, 2),   # Goiania
        (-20.5, -54.6, 2),   # Campo Grande
        (-15.6, -56.1, 2),   # Cuiaba
        # Sudeste
        (-23.5, -46.6, 3),   # Sao Paulo
        (-22.9, -43.2, 3),   # Rio de Janeiro
        (-19.9, -43.9, 3),   # Belo Horizonte
        (-20.3, -40.3, 3),   # Vitoria
        # Sul
        (-25.4, -49.3, 4),   # Curitiba
        (-30.0, -51.2, 4),   # Porto Alegre
        (-27.6, -48.5, 4),   # Florianopolis
    ]
    arr = np.array(caps, dtype=float)
    lat, lon, y = arr[:, 0], arr[:, 1], arr[:, 2].astype(int)
    # X = longitude (west→east), Y = latitude (south→north)
    X = np.column_stack([lon, lat])
    X = _norm(X)
    # invert Y so north (high lat) is at top of canvas
    X = np.column_stack([X[:, 0], 1 - X[:, 1]])
    return _pack(X, y)

# ── Registry ──────────────────────────────────────────────────────────────────

# ── Empty (user draws points) ────────────────────────────────────────────────

def vazio():
    """Empty dataset — returns no points so the user can draw their own."""
    return []

DATASETS = {
    "vazio": {
        "label":       "Vazio",
        "description": "Desenhe os pontos",
        "fn":          vazio,
        "classes":     ["Classe A", "Classe B", "Classe C", "Classe D"],
    },
    "gaussiano": {
        "label":       "Gaussiano",
        "description": "2 grupos separaveis",
        "fn":          gaussian,
        "classes":     ["Classe A", "Classe B"],
    },
    "xor": {
        "label":       "XOR",
        "description": "4 cantos, 2 classes",
        "fn":          xor,
        "classes":     ["Classe A", "Classe B"],
    },
    "circulos": {
        "label":       "Circulos",
        "description": "Aneis concentricos",
        "fn":          circles,
        "classes":     ["Interno", "Externo"],
    },
    "espiral": {
        "label":       "Espiral",
        "description": "Dois bracos em espiral",
        "fn":          spiral,
        "classes":     ["Braco A", "Braco B"],
    },
    "quatro": {
        "label":       "4 Grupos",
        "description": "Quatro cantos, 4 classes",
        "fn":          four_gaussians,
        "classes":     ["G1", "G2", "G3", "G4"],
    },
    "iris": {
        "label":       "Iris",
        "description": "Dataset real — 150 flores",
        "fn":          iris,
        "classes":     ["Setosa", "Versicolor", "Virginica"],
    },
    "brasil": {
        "label":       "Brasil",
        "description": "Capitais por regiao",
        "fn":          brasil,
        "classes":     ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"],
    },
}


def get_dataset(key: str) -> dict:
    entry = DATASETS[key]
    points = entry["fn"]()
    return {
        "id":          key,
        "label":       entry["label"],
        "description": entry["description"],
        "classes":     entry["classes"],
        "points":      points,
    }


def list_datasets() -> list[dict]:
    result = []
    for key, entry in DATASETS.items():
        pts = entry["fn"]()
        result.append({
            "id":          key,
            "label":       entry["label"],
            "description": entry["description"],
            "classes":     entry["classes"],
            "points":      pts,           # thumbnail uses these
        })
    return result
