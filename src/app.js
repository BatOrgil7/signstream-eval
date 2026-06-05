import { DrawingUtils, FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const DEFAULT_LABELS = ["A", "B", "C", "I", "L", "Y"];
const STORAGE_KEY = "few-shot-sign-lab-session-v1";
const CAPTURE_INTERVAL_MS = 140;
const SMOOTHING_WINDOW = 7;
const MEDIAPIPE_WASM_URL = "/wasm";
const HAND_MODEL_URL = "/models/hand_landmarker.task";

const state = {
  labels: [...DEFAULT_LABELS],
  activeLabel: "A",
  samples: [],
  stream: null,
  handLandmarker: null,
  drawingUtils: null,
  handConnections: null,
  recording: false,
  latestVector: null,
  latestRawLandmarks: null,
  latestPrediction: null,
  lastVideoTime: -1,
  lastCaptureAt: 0,
  predictions: [],
  totalPredictions: 0,
  confidentPredictions: 0,
};

const els = {
  video: document.querySelector("#webcam"),
  canvas: document.querySelector("#overlay-canvas"),
  modelStatus: document.querySelector("#model-status"),
  modelStatusDot: document.querySelector("#model-status-dot"),
  cameraButton: document.querySelector("#camera-button"),
  resetSessionButton: document.querySelector("#reset-session-button"),
  labelInput: document.querySelector("#label-input"),
  labelCount: document.querySelector("#label-count"),
  signGrid: document.querySelector("#sign-grid"),
  activeLabel: document.querySelector("#active-label"),
  activeSampleCount: document.querySelector("#active-sample-count"),
  totalSampleCount: document.querySelector("#total-sample-count"),
  captureOnceButton: document.querySelector("#capture-once-button"),
  recordButton: document.querySelector("#record-button"),
  clearLabelButton: document.querySelector("#clear-label-button"),
  neighborSlider: document.querySelector("#neighbor-slider"),
  neighborValue: document.querySelector("#neighbor-value"),
  thresholdSlider: document.querySelector("#threshold-slider"),
  thresholdValue: document.querySelector("#threshold-value"),
  prediction: document.querySelector("#prediction"),
  confidence: document.querySelector("#confidence"),
  trialSummary: document.querySelector("#trial-summary"),
  exportJsonButton: document.querySelector("#export-json-button"),
  exportCsvButton: document.querySelector("#export-csv-button"),
};

const ctx = els.canvas.getContext("2d");

init();

async function init() {
  restoreSession();
  bindEvents();
  renderLabels();
  renderCounts();
  updateRecognizerControls();
  setStatus("Loading hand tracker", "loading");

  try {
    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);

    state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: HAND_MODEL_URL,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.55,
    });

    state.drawingUtils = new DrawingUtils(ctx);
    state.handConnections = HandLandmarker.HAND_CONNECTIONS;
    setStatus("Hand tracker ready", "ready");
  } catch (error) {
    console.error(error);
    setStatus("Hand tracker failed to load", "error");
  }
}

function bindEvents() {
  els.cameraButton.addEventListener("click", toggleCamera);
  els.resetSessionButton.addEventListener("click", resetSession);
  els.captureOnceButton.addEventListener("click", () => captureSample(Date.now(), true));
  els.recordButton.addEventListener("click", toggleRecording);
  els.clearLabelButton.addEventListener("click", clearActiveLabelSamples);
  els.exportJsonButton.addEventListener("click", exportJson);
  els.exportCsvButton.addEventListener("click", exportCsv);

  els.labelInput.addEventListener("change", () => {
    const nextLabels = parseLabels(els.labelInput.value);
    if (!nextLabels.length) {
      els.labelInput.value = state.labels.join(", ");
      return;
    }

    state.labels = nextLabels;
    if (!state.labels.includes(state.activeLabel)) {
      state.activeLabel = state.labels[0];
    }
    state.samples = state.samples.filter((sample) => state.labels.includes(sample.label));
    persistSession();
    renderLabels();
    renderCounts();
  });

  els.neighborSlider.addEventListener("input", updateRecognizerControls);
  els.thresholdSlider.addEventListener("input", updateRecognizerControls);
}

function parseLabels(value) {
  return [
    ...new Set(
      value
        .split(",")
        .map((label) => label.trim())
        .filter(Boolean)
        .slice(0, 24),
    ),
  ];
}

async function toggleCamera() {
  if (state.stream) {
    stopCamera();
    return;
  }

  if (!state.handLandmarker) {
    setStatus("Hand tracker is not ready", "error");
    return;
  }

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    els.video.srcObject = state.stream;
    await els.video.play();
    els.cameraButton.textContent = "Stop Camera";
    setStatus("Camera running", "ready");
    requestAnimationFrame(processFrame);
  } catch (error) {
    console.error(error);
    setStatus("Camera permission blocked", "error");
  }
}

function stopCamera() {
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  state.recording = false;
  els.video.srcObject = null;
  els.cameraButton.textContent = "Start Camera";
  els.recordButton.textContent = "Record Stream";
  clearCanvas();
  setPrediction(null);
  setStatus("Camera stopped", state.handLandmarker ? "ready" : "loading");
}

function processFrame(now) {
  if (!state.stream || !state.handLandmarker) {
    return;
  }

  if (els.video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = els.video.currentTime;
    resizeCanvas();

    const result = state.handLandmarker.detectForVideo(els.video, now);
    drawResult(result);

    const landmarks = result.landmarks?.[0];
    if (landmarks) {
      state.latestRawLandmarks = landmarks;
      state.latestVector = normalizeLandmarks(landmarks);
      const prediction = predict(state.latestVector);
      setPrediction(prediction);

      if (state.recording) {
        captureSample(now, false);
      }
    } else {
      state.latestRawLandmarks = null;
      state.latestVector = null;
      setPrediction(null);
    }
  }

  requestAnimationFrame(processFrame);
}

function resizeCanvas() {
  const rect = els.video.getBoundingClientRect();
  const width = Math.round(rect.width);
  const height = Math.round(rect.height);
  if (els.canvas.width !== width || els.canvas.height !== height) {
    els.canvas.width = width;
    els.canvas.height = height;
  }
}

function drawResult(result) {
  clearCanvas();
  if (!result.landmarks?.length) {
    return;
  }

  for (const landmarks of result.landmarks) {
    state.drawingUtils.drawConnectors(landmarks, state.handConnections, {
      color: "#40c4d1",
      lineWidth: 3,
    });
    state.drawingUtils.drawLandmarks(landmarks, {
      color: "#f6c453",
      fillColor: "#ffffff",
      lineWidth: 2,
      radius: 4,
    });
  }
}

function clearCanvas() {
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
}

function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const scale =
    distance3d(wrist, middleMcp) ||
    Math.max(...landmarks.map((point) => distance3d(wrist, point))) ||
    1;

  return landmarks.flatMap((point) => [
    (point.x - wrist.x) / scale,
    (point.y - wrist.y) / scale,
    (point.z - wrist.z) / scale,
  ]);
}

function distance3d(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function captureSample(now, force) {
  if (!state.latestVector) {
    setStatus("Show one hand before capturing", "error");
    return;
  }

  if (!force && now - state.lastCaptureAt < CAPTURE_INTERVAL_MS) {
    return;
  }

  state.lastCaptureAt = now;
  state.samples.push({
    label: state.activeLabel,
    vector: [...state.latestVector],
    capturedAt: new Date().toISOString(),
  });
  persistSession();
  renderCounts();
  setStatus(`Captured ${state.activeLabel}`, "ready");
}

function predict(vector) {
  if (!vector || state.samples.length < 2) {
    return null;
  }

  const k = Math.min(Number(els.neighborSlider.value), state.samples.length);
  const nearest = state.samples
    .map((sample) => ({
      label: sample.label,
      distance: euclideanDistance(vector, sample.vector),
    }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, k);

  const weights = new Map();
  for (const item of nearest) {
    const weight = 1 / Math.max(item.distance, 0.0001);
    weights.set(item.label, (weights.get(item.label) || 0) + weight);
  }

  const totalWeight = [...weights.values()].reduce((sum, value) => sum + value, 0);
  const [label, weight] = [...weights.entries()].sort((a, b) => b[1] - a[1])[0];
  const confidence = totalWeight ? weight / totalWeight : 0;

  state.predictions.push(label);
  state.predictions = state.predictions.slice(-SMOOTHING_WINDOW);

  const smoothedLabel = mostFrequent(state.predictions);
  const smoothedConfidence =
    state.predictions.filter((item) => item === smoothedLabel).length /
    state.predictions.length;

  return {
    label: smoothedLabel,
    confidence: Math.min(confidence * 0.68 + smoothedConfidence * 0.32, 1),
  };
}

function euclideanDistance(a, b) {
  let sum = 0;
  for (let index = 0; index < a.length; index += 1) {
    const diff = a[index] - b[index];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function mostFrequent(items) {
  const counts = new Map();
  for (const item of items) {
    counts.set(item, (counts.get(item) || 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
}

function setPrediction(prediction) {
  const threshold = Number(els.thresholdSlider.value) / 100;
  state.latestPrediction = prediction;

  if (!prediction) {
    els.prediction.textContent = "No hand";
    els.confidence.textContent = state.samples.length
      ? "Waiting for landmarks"
      : "Capture samples first";
    return;
  }

  state.totalPredictions += 1;
  const confident = prediction.confidence >= threshold;
  if (confident) {
    state.confidentPredictions += 1;
  }

  els.prediction.textContent = confident ? prediction.label : "Unsure";
  els.confidence.textContent = `${Math.round(prediction.confidence * 100)}% confidence`;
  els.trialSummary.textContent = `${state.confidentPredictions}/${state.totalPredictions} confident predictions this session.`;
}

function toggleRecording() {
  state.recording = !state.recording;
  els.recordButton.textContent = state.recording ? "Stop Recording" : "Record Stream";
  setStatus(state.recording ? `Recording ${state.activeLabel}` : "Recording stopped", "ready");
}

function clearActiveLabelSamples() {
  state.samples = state.samples.filter((sample) => sample.label !== state.activeLabel);
  persistSession();
  renderCounts();
}

function resetSession() {
  state.samples = [];
  state.predictions = [];
  state.totalPredictions = 0;
  state.confidentPredictions = 0;
  state.latestPrediction = null;
  localStorage.removeItem(STORAGE_KEY);
  renderCounts();
  setPrediction(null);
  setStatus("Session reset", "ready");
}

function renderLabels() {
  els.labelInput.value = state.labels.join(", ");
  els.labelCount.textContent = `${state.labels.length} signs`;
  els.signGrid.replaceChildren(
    ...state.labels.map((label) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `sign-button${label === state.activeLabel ? " active" : ""}`;
      button.textContent = label;
      button.addEventListener("click", () => {
        state.activeLabel = label;
        renderLabels();
        renderCounts();
      });
      return button;
    }),
  );
  els.activeLabel.textContent = state.activeLabel;
}

function renderCounts() {
  els.activeSampleCount.textContent = state.samples.filter(
    (sample) => sample.label === state.activeLabel,
  ).length;
  els.totalSampleCount.textContent = state.samples.length;
}

function updateRecognizerControls() {
  els.neighborValue.textContent = els.neighborSlider.value;
  els.thresholdValue.textContent = `${els.thresholdSlider.value}%`;
}

function setStatus(message, type) {
  els.modelStatus.textContent = message;
  els.modelStatusDot.classList.remove("ready", "error");
  if (type === "ready") {
    els.modelStatusDot.classList.add("ready");
  }
  if (type === "error") {
    els.modelStatusDot.classList.add("error");
  }
}

function persistSession() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      labels: state.labels,
      activeLabel: state.activeLabel,
      samples: state.samples,
    }),
  );
}

function restoreSession() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return;
  }

  try {
    const saved = JSON.parse(raw);
    if (Array.isArray(saved.labels) && saved.labels.length) {
      state.labels = saved.labels;
    }
    if (state.labels.includes(saved.activeLabel)) {
      state.activeLabel = saved.activeLabel;
    }
    if (Array.isArray(saved.samples)) {
      state.samples = saved.samples.filter(
        (sample) =>
          state.labels.includes(sample.label) &&
          Array.isArray(sample.vector) &&
          sample.vector.length === 63,
      );
    }
  } catch {
    localStorage.removeItem(STORAGE_KEY);
  }
}

function exportJson() {
  const payload = {
    project: "Few-Shot Sign Lab",
    exportedAt: new Date().toISOString(),
    labels: state.labels,
    sampleCount: state.samples.length,
    samples: state.samples,
    recognizer: {
      type: "k-nearest-neighbors",
      neighbors: Number(els.neighborSlider.value),
      confidenceThreshold: Number(els.thresholdSlider.value) / 100,
      featureSpace: "MediaPipe hand landmarks normalized by wrist and middle MCP distance",
    },
  };
  downloadFile("few-shot-sign-session.json", JSON.stringify(payload, null, 2), "application/json");
}

function exportCsv() {
  const header = ["label", "captured_at", ...Array.from({ length: 63 }, (_, i) => `feature_${i}`)];
  const rows = state.samples.map((sample) => [
    sample.label,
    sample.capturedAt,
    ...sample.vector.map((value) => Number(value).toFixed(8)),
  ]);
  const csv = [header, ...rows]
    .map((row) => row.map((cell) => `"${String(cell).replaceAll('"', '""')}"`).join(","))
    .join("\n");
  downloadFile("few-shot-sign-samples.csv", csv, "text/csv");
}

function downloadFile(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
