import './style.css';

const CANVAS_SIZE = 720;
const COLORS = [
  { name: 'Azul', hex: '#3f7396', rgb: [63, 115, 150] },
  { name: 'Vermelho', hex: '#bd5b52', rgb: [189, 91, 82] },
  { name: 'Verde', hex: '#5d8866', rgb: [93, 136, 102] },
  { name: 'Amarelo', hex: '#c18a36', rgb: [193, 138, 54] },
  { name: 'Roxo', hex: '#846c9c', rgb: [132, 108, 156] },
];

const secure = window.location.protocol === 'https:';
const API = window.location.origin;
const WS = `${secure ? 'wss' : 'ws'}://${window.location.host}/ws/train`;

const elements = {
  dataset: document.querySelector('#dataset'),
  model: document.querySelector('#model'),
  svmControls: document.querySelector('#svm-controls'),
  kernel: document.querySelector('#kernel'),
  cValue: document.querySelector('#c-value'),
  gamma: document.querySelector('#gamma'),
  gammaRow: document.querySelector('#gamma-row'),
  degree: document.querySelector('#degree'),
  degreeRow: document.querySelector('#degree-row'),
  classButtons: document.querySelector('#class-buttons'),
  train: document.querySelector('#train'),
  reset: document.querySelector('#reset'),
  clear: document.querySelector('#clear'),
  boundaryCanvas: document.querySelector('#boundary-canvas'),
  pointsCanvas: document.querySelector('#points-canvas'),
  status: document.querySelector('#status'),
  epoch: document.querySelector('#epoch'),
  loss: document.querySelector('#loss'),
  accuracy: document.querySelector('#accuracy'),
  legend: document.querySelector('#legend'),
};

const state = {
  datasets: [],
  points: [],
  currentColor: 0,
  classNames: ['Classe A', 'Classe B', 'Classe C', 'Classe D'],
  selectedDataset: null,
  training: false,
};

const boundaryContext = elements.boundaryCanvas.getContext('2d');
const pointsContext = elements.pointsCanvas.getContext('2d');
const offscreen = document.createElement('canvas');

function setStatus(message, kind = '') {
  elements.status.textContent = message;
  elements.status.dataset.kind = kind;
}

function resetStats() {
  elements.epoch.textContent = '—';
  elements.loss.textContent = '—';
  elements.accuracy.textContent = '—';
}

function clearBoundary() {
  boundaryContext.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
}

function setTraining(training) {
  state.training = training;
  document.querySelectorAll('[data-lock]').forEach((element) => {
    element.disabled = training;
  });
  elements.train.textContent = training ? 'Treinando…' : 'Treinar';
  renderClasses();
}

function renderPoints() {
  pointsContext.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  state.points.forEach((point) => {
    const color = COLORS[point.color] || COLORS[0];
    pointsContext.beginPath();
    pointsContext.arc(point.x, point.y, 7, 0, Math.PI * 2);
    pointsContext.fillStyle = color.hex;
    pointsContext.fill();
    pointsContext.lineWidth = 2;
    pointsContext.strokeStyle = '#ffffff';
    pointsContext.stroke();
  });
}

function renderBoundary(grid, gridSize) {
  offscreen.width = gridSize;
  offscreen.height = gridSize;
  const context = offscreen.getContext('2d');
  const image = context.createImageData(gridSize, gridSize);

  grid.forEach((classIndex, index) => {
    const [r, g, b] = (COLORS[classIndex] || COLORS[0]).rgb;
    image.data[index * 4] = r;
    image.data[index * 4 + 1] = g;
    image.data[index * 4 + 2] = b;
    image.data[index * 4 + 3] = 115;
  });

  context.putImageData(image, 0, 0);
  boundaryContext.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  boundaryContext.imageSmoothingEnabled = true;
  boundaryContext.imageSmoothingQuality = 'high';
  boundaryContext.drawImage(offscreen, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
}

function renderClasses() {
  elements.classButtons.replaceChildren();
  elements.legend.replaceChildren();

  state.classNames.slice(0, COLORS.length).forEach((name, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'class-button';
    button.dataset.active = String(index === state.currentColor);
    button.disabled = state.training;
    button.innerHTML = `<span style="background:${COLORS[index].hex}"></span>${name}`;
    button.addEventListener('click', () => {
      state.currentColor = index;
      renderClasses();
    });
    elements.classButtons.append(button);

    const item = document.createElement('div');
    item.innerHTML = `<span style="background:${COLORS[index].hex}"></span>${name}`;
    elements.legend.append(item);
  });
}

function loadDataset(dataset) {
  state.selectedDataset = dataset.id;
  state.classNames = dataset.classes;
  state.currentColor = 0;
  state.points = dataset.points.map((point) => ({
    x: point.x * CANVAS_SIZE,
    y: point.y * CANVAS_SIZE,
    color: point.color,
  }));
  elements.dataset.value = dataset.id;
  clearBoundary();
  resetStats();
  renderPoints();
  renderClasses();
  setStatus('Pronto');
}

function resetDataset() {
  const dataset = state.datasets.find((item) => item.id === state.selectedDataset);
  if (dataset) loadDataset(dataset);
}

function clearPoints() {
  state.points = [];
  clearBoundary();
  resetStats();
  renderPoints();
  setStatus('Canvas limpo');
}

function canvasCoordinates(event) {
  const bounds = elements.pointsCanvas.getBoundingClientRect();
  return {
    x: (event.clientX - bounds.left) * CANVAS_SIZE / bounds.width,
    y: (event.clientY - bounds.top) * CANVAS_SIZE / bounds.height,
  };
}

function addPoint(event) {
  if (state.training) return;
  const { x, y } = canvasCoordinates(event);
  state.points.push({ x, y, color: state.currentColor });
  clearBoundary();
  resetStats();
  renderPoints();
}

function removePoint(event) {
  event.preventDefault();
  if (state.training || !state.points.length) return;
  const { x, y } = canvasCoordinates(event);
  let nearest = -1;
  let distance = 22;

  state.points.forEach((point, index) => {
    const currentDistance = Math.hypot(point.x - x, point.y - y);
    if (currentDistance < distance) {
      nearest = index;
      distance = currentDistance;
    }
  });

  if (nearest >= 0) {
    state.points.splice(nearest, 1);
    clearBoundary();
    resetStats();
    renderPoints();
  }
}

function updateModelControls() {
  elements.svmControls.hidden = elements.model.value !== 'svm';
}

function updateKernelControls() {
  const kernel = elements.kernel.value;
  elements.gammaRow.hidden = kernel === 'linear';
  elements.degreeRow.hidden = kernel !== 'poly';
}

function svmParameters() {
  const gammaText = elements.gamma.value.trim();
  const gamma = ['scale', 'auto'].includes(gammaText) ? gammaText : Number(gammaText);

  if (!['scale', 'auto'].includes(gammaText) && (!Number.isFinite(gamma) || gamma <= 0)) {
    throw new Error('Gamma deve ser scale, auto ou um número positivo.');
  }

  return {
    kernel: elements.kernel.value,
    C: Number(elements.cValue.value),
    gamma,
    degree: Number(elements.degree.value),
  };
}

function train() {
  const classes = new Set(state.points.map((point) => point.color));
  if (state.points.length < 2 || classes.size < 2) {
    setStatus('Use ao menos duas classes.', 'error');
    return;
  }

  let modelParams = {};
  try {
    if (elements.model.value === 'svm') modelParams = svmParameters();
  } catch (error) {
    setStatus(error.message, 'error');
    return;
  }

  setTraining(true);
  clearBoundary();
  resetStats();
  setStatus('Treinando…');

  const socket = new WebSocket(WS);
  socket.addEventListener('open', () => {
    socket.send(JSON.stringify({
      points: state.points.map((point) => ({
        x: point.x / CANVAS_SIZE,
        y: point.y / CANVAS_SIZE,
        color: point.color,
      })),
      model_type: elements.model.value,
      model_params: modelParams,
    }));
  });

  socket.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);

    if (message.type === 'epoch') {
      elements.epoch.textContent = message.epoch;
      elements.loss.textContent = message.loss.toFixed(4);
      elements.accuracy.textContent = `${(message.accuracy * 100).toFixed(1)}%`;
      renderBoundary(message.grid, message.grid_size);
    }

    if (message.type === 'done') {
      setTraining(false);
      renderClasses();
      setStatus('Treino concluído', 'success');
      socket.close();
    }

    if (message.type === 'error') {
      setTraining(false);
      renderClasses();
      setStatus(message.msg, 'error');
      socket.close();
    }
  });

  socket.addEventListener('error', () => {
    setTraining(false);
    setStatus('Não foi possível acessar a API.', 'error');
  });

  socket.addEventListener('close', () => {
    if (state.training) {
      setTraining(false);
      setStatus('Conexão encerrada antes do fim do treino.', 'error');
    }
  });
}

async function loadDatasets() {
  try {
    const response = await fetch(`${API}/api/datasets`);
    if (!response.ok) throw new Error('API indisponível');
    state.datasets = await response.json();
    elements.dataset.replaceChildren();

    state.datasets.forEach((dataset) => {
      const option = document.createElement('option');
      option.value = dataset.id;
      option.textContent = `${dataset.label} — ${dataset.description}`;
      elements.dataset.append(option);
    });

    loadDataset(state.datasets.find((dataset) => dataset.id === 'gaussiano') || state.datasets[0]);
  } catch {
    setStatus('Inicie a API na porta 8000.', 'error');
    elements.dataset.innerHTML = '<option>API indisponível</option>';
  }
}

elements.dataset.addEventListener('change', () => {
  const dataset = state.datasets.find((item) => item.id === elements.dataset.value);
  if (dataset) loadDataset(dataset);
});
elements.model.addEventListener('change', updateModelControls);
elements.kernel.addEventListener('change', updateKernelControls);
elements.train.addEventListener('click', train);
elements.reset.addEventListener('click', resetDataset);
elements.clear.addEventListener('click', clearPoints);
elements.pointsCanvas.addEventListener('click', addPoint);
elements.pointsCanvas.addEventListener('contextmenu', removePoint);

updateModelControls();
updateKernelControls();
renderClasses();
loadDatasets();
