import io
import os
 
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
from ultralytics import YOLO
 
HERE = os.path.dirname(os.path.abspath(__file__))
MAX_BYTES = 15 * 1024 * 1024
MAX_SIDE = 1280
 
app = FastAPI(title="Backyard Sweep")
model = YOLO(os.path.join(HERE, "best.pt"))
 
 
@app.get("/")
def home():
    return FileResponse(os.path.join(HERE, "index.html"))
 
 
@app.get("/health")
def health():
    return {"ok": True}
 
 
# Plain `def` (not async) so FastAPI runs the heavy model call in a worker thread.
@app.post("/api/detect")
def detect(file: UploadFile = File(...), conf: float = Form(0.10)):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="That photo is too large. Use one under 15 MB.")
 
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Couldn't open that photo. Try a JPG or PNG.")
 
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    w, h = img.size
    conf = min(max(conf, 0.05), 0.95)
 
    result = model.predict(img, conf=conf, imgsz=640, verbose=False)[0]
 
    detections = []
    for (x1, y1, x2, y2), score, cls in zip(
        result.boxes.xyxy.tolist(),
        result.boxes.conf.tolist(),
        result.boxes.cls.tolist(),
    ):
        detections.append(
            {
                "label": result.names[int(cls)],
                "conf": round(score, 3),
                # normalised 0-1 so the page can draw them at any size
                "box": [x1 / w, y1 / h, x2 / w, y2 / h],
            }
        )
 
    return {"width": w, "height": h, "detections": detections}
