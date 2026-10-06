import { useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const API = "http://localhost:8000/verify";
const CROP_SCALE = 1.5; // same face margin used when the training crops were made

const colors = { bg: "#12151A", card: "#1E232B", text: "#F6F5F1", dim: "#B7BDC4", live: "#2DD4BF", spoof: "#F5A623" };

function ResultCard({ title, r }) {
  if (!r) return null;
  const good = r.is_live;
  return (
    <div style={{ background: colors.card, borderRadius: 12, padding: 20, minWidth: 220 }}>
      <div style={{ color: colors.dim, fontSize: 14, marginBottom: 6 }}>{title}</div>
      <div style={{ fontSize: 28, fontWeight: 700, color: good ? colors.live : colors.spoof }}>
        {good ? "LIVE" : "SPOOF"}
      </div>
      <div style={{ color: colors.text, marginTop: 4 }}>confidence {(r.confidence * 100).toFixed(1)}%</div>
    </div>
  );
}

export default function App() {
  const videoRef = useRef(null);
  const landmarkerRef = useRef(null);
  const [status, setStatus] = useState("Loading face detector...");
  const [result, setResult] = useState(null);
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let stream;
    (async () => {
      try {
        const files = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
        );
        landmarkerRef.current = await FaceLandmarker.createFromOptions(files, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          },
          runningMode: "VIDEO",
          numFaces: 1,
        });
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoRef.current.srcObject = stream;
        setStatus("Ready. Look at the camera and press Verify, or upload an image.");
      } catch (e) {
        setStatus("Setup problem: " + e.message + " (webcam allowed? internet on for the face model?)");
      }
    })();
    return () => stream && stream.getTracks().forEach((t) => t.stop());
  }, []);

  // Crop a square around the detected face (with margin) and return a 224x224 PNG data URL.
  function cropFace() {
    const video = videoRef.current;
    const res = landmarkerRef.current.detectForVideo(video, performance.now());
    const lm = res.faceLandmarks && res.faceLandmarks[0];
    if (!lm) return null;
    const W = video.videoWidth, H = video.videoHeight;
    const xs = lm.map((p) => p.x * W), ys = lm.map((p) => p.y * H);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const side = Math.max(maxX - minX, maxY - minY) * CROP_SCALE;
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 224;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, 224, 224);
    ctx.drawImage(video, cx - side / 2, cy - side / 2, side, side, 0, 0, 224, 224);
    return canvas.toDataURL("image/png");
  }

  async function send(dataUrl) {
    setBusy(true);
    setPreview(dataUrl);
    try {
      const res = await fetch(API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64: dataUrl }),
      });
      setResult(await res.json());
      setStatus("Done.");
    } catch (e) {
      setResult(null);
      setStatus("Backend not reachable. Is uvicorn running on port 8000?");
    }
    setBusy(false);
  }

  function onVerify() {
    if (!landmarkerRef.current) return;
    const url = cropFace();
    if (!url) return setStatus("No face found. Move closer and try again.");
    send(url);
  }

  function onUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => send(reader.result);
    reader.readAsDataURL(file);
  }

  const models = result && result.models;
  return (
    <div style={{ fontFamily: "Inter, system-ui, sans-serif", color: colors.text, padding: 32, background: colors.bg, minHeight: "100vh" }}>
      <h1 style={{ margin: 0 }}>Kiosk Guard</h1>
      <p style={{ color: colors.dim }}>Webcam liveness check with a baseline and a hardened detector, side by side.</p>
      <div style={{ display: "flex", gap: 32, flexWrap: "wrap", alignItems: "flex-start" }}>
        <div>
          <video ref={videoRef} autoPlay playsInline muted width={480} height={360}
                 style={{ borderRadius: 12, background: "#000", transform: "scaleX(-1)" }} />
          <div style={{ marginTop: 12, display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <button onClick={onVerify} disabled={busy}
                    style={{ padding: "10px 18px", fontSize: 16, borderRadius: 8, border: "none", background: colors.live, cursor: "pointer" }}>
              {busy ? "Checking..." : "Verify live face"}
            </button>
            <label style={{ color: colors.dim, fontSize: 14 }}>
              or upload an image: <input type="file" accept="image/png,image/jpeg" onChange={onUpload} />
            </label>
          </div>
          <div style={{ color: colors.dim, marginTop: 8, fontSize: 14 }}>{status}</div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {preview && <img src={preview} alt="what the model saw" width={224} height={224} style={{ borderRadius: 12 }} />}
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            {models && models.baseline && <ResultCard title="Baseline model" r={models.baseline} />}
            {models && <ResultCard title="Hardened model" r={models.hardened || result} />}
          </div>
        </div>
      </div>
    </div>
  );
}
