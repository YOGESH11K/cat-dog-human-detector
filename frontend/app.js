const video = document.getElementById("video");
const statusEl = document.getElementById("status");
const result = document.getElementById("result");

const EMOJI = { cat: "🐱", dog: "🐶", human: "🧑" };
const DETECT_INTERVAL_MS = 800;

let running = false;
let busy = false;

async function initCamera() {
  statusEl.textContent = "Requesting camera access...";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    statusEl.textContent = "Live detection running - point at a cat, dog, or human";
    running = true;
    loop();
  } catch (err) {
    statusEl.textContent = "Camera unavailable: " + err.message;
  }
}

function captureFrame() {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  return canvas.toDataURL("image/jpeg", 0.7);
}

async function detect(imageDataUrl) {
  try {
    const res = await fetch(API_BASE + "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageDataUrl }),
    });
    if (!res.ok) return null;
    const data = await res.json();
    if (data && data.prediction) {
      render(data);
      statusEl.textContent = "Sees: " + data.prediction;
    }
  } catch (err) {
    /* transient network error - keep last result */
  }
}

async function loop() {
  if (busy) return;
  busy = true;
  if (running && video.readyState >= 2) {
    await detect(captureFrame());
  }
  busy = false;
  if (running) setTimeout(loop, DETECT_INTERVAL_MS);
}

function render(data) {
  const label = data.prediction;
  const conf = Math.round(data.confidence * 100);
  result.innerHTML = `
    <div class="emote">${EMOJI[label] || "❔"}</div>
    <div class="prediction">${label}</div>
    <div class="confidence">${conf}% confident</div>
    <div class="track"><div class="track-fill" style="width:${conf}%"></div></div>
    <div class="fact">${data.fun_fact || ""}</div>
  `;
}

initCamera();