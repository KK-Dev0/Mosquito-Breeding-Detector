import os
import gradio as gr
from ultralytics import YOLO

model = YOLO("best.pt")

def detect(img, conf):
    r = model.predict(img, conf=conf, imgsz=640)[0]
    annotated = r.plot()[..., ::-1]
    counts = {}
    for c in r.boxes.cls.tolist():
        name = r.names[int(c)]
        counts[name] = counts.get(name, 0) + 1
    summary = ", ".join(f"{v} x {k}" for k, v in counts.items()) or "No breeding sites detected"
    return annotated, summary

demo = gr.Interface(
    fn=detect,
    inputs=[gr.Image(type="pil", label="Upload a photo"),
            gr.Slider(0.1, 0.9, value=0.25, label="Confidence threshold")],
    outputs=[gr.Image(label="Detections"), gr.Textbox(label="Potential breeding sites found")],
    title="Backyard Sweep: Mosquito Breeding Site Detector",
    description="YOLOv8n trained on VisText-Mosquito (CC BY 4.0, Sayeedi Faiyaz & Jisan, DOI: 10.17632/rtsfh7jh7p.3). Detects: Bottle, Coconut-Exocarp, Drain-Inlet, Tire, Vase.",
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
