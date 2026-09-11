const video = document.getElementById("video");
const captureBtn = document.getElementById("captureBtn");
const result = document.getElementById("result");
const statusEl = document.getElementById("status");

const EMOJI = { cat: "🐱", dog: "🐶", human: "🧑" };

async function initCamera() {
  statusEl.textContent = "Requesting camera access…";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    statusEl.textContent = "Camera ready. Point it at a cat, dog, or person.";
    captureBtn.disabled = false;
  } catch (err) {
    statusEl.textContent = "Camera unavailable: " + err.message;
    captureBtn.disabled = true;
  }
}

function captureFrame() {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  return canvas.toDataURL("image/jpeg", 0.85);
}

async function detect(imageDataUrl) {
  const res = await fetch(API_BASE + "/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imageDataUrl }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || ("HTTP " + res.status));
  }
  return res.json();
}

function render(result, data) {
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

captureBtn.addEventListener("click", async () => {
  captureBtn.disabled = true;
  statusEl.textContent = "Detecting…";
  try {
    const data = await detect(captureFrame());
    render(result, data);
    statusEl.textContent = "Detected: " + data.prediction;
  } catch (err) {
    result.innerHTML = `<div class="error">Detection failed: ${err.message}</div>`;
    statusEl.textContent = err.message;
  } finally {
    captureBtn.disabled = false;
  }
});

initCamera();