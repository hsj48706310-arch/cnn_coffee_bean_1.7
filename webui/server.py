from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent

DETECT_MODEL_PATHS = {
    "yolo26n": ROOT_DIR / "artifacts" / "train" / "coffee_yolo26n_e30_img832" / "weights" / "best.pt",
    "yolo26s": ROOT_DIR / "artifacts" / "train" / "coffee_yolo26s_e5best_plus15_gpu" / "weights" / "best.pt",
}
ROAST_CHECKPOINT_PATH = ROOT_DIR / "artifacts" / "roast_cls_train_e5" / "best.pt"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
_detector_cache: dict[str, YOLO] = {}
_roast_cache: dict[str, Any] | None = None


def _encode_image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _load_detector(model_key: str) -> YOLO:
    if model_key in _detector_cache:
        return _detector_cache[model_key]

    model_path = DETECT_MODEL_PATHS[model_key]
    if not model_path.exists():
        raise FileNotFoundError(f"Detection model not found: {model_path}")

    detector = YOLO(str(model_path))
    _detector_cache[model_key] = detector
    return detector


def _load_roast_model() -> dict[str, Any]:
    global _roast_cache
    if _roast_cache is not None:
        return _roast_cache

    if not ROAST_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Roasting model not found: {ROAST_CHECKPOINT_PATH}")

    checkpoint = torch.load(ROAST_CHECKPOINT_PATH, map_location=DEVICE)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    img_size = int(checkpoint.get("img_size", 224))

    model = efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_to_idx))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    _roast_cache = {
        "model": model,
        "transform": transform,
        "idx_to_class": idx_to_class,
    }
    return _roast_cache


def _run_detection(image: Image.Image, model_key: str) -> dict[str, Any]:
    detector = _load_detector(model_key)
    results = detector.predict(
        source=image,
        verbose=False,
        device=0 if DEVICE.startswith("cuda") else "cpu",
    )
    result = results[0]

    detections: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}

    for box in result.boxes:
        cls_idx = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_name = result.names[cls_idx]

        detections.append(
            {
                "class_name": class_name,
                "confidence": round(conf, 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            }
        )
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    plotted_bgr = result.plot()
    plotted_rgb = Image.fromarray(plotted_bgr[..., ::-1])

    return {
        "counts": class_counts,
        "detections": detections,
        "annotated_image": _encode_image_to_data_url(plotted_rgb),
    }


def _run_roast_classification(image: Image.Image) -> dict[str, Any]:
    roast_bundle = _load_roast_model()
    model = roast_bundle["model"]
    transform = roast_bundle["transform"]
    idx_to_class = roast_bundle["idx_to_class"]

    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_idx = int(torch.argmax(probs).item())
    top_conf = float(probs[top_idx].item())
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)

    top2: list[dict[str, Any]] = []
    for rank in range(min(2, len(sorted_idx))):
        idx = int(sorted_idx[rank].item())
        top2.append(
            {
                "class_name": idx_to_class[idx],
                "confidence": round(float(sorted_probs[rank].item()), 4),
            }
        )

    return {
        "predicted_stage": idx_to_class[top_idx],
        "confidence": round(top_conf, 4),
        "top2": top2,
    }


@app.get("/")
def serve_index() -> Any:
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/<path:asset_path>")
def serve_static(asset_path: str) -> Any:
    return send_from_directory(WEB_DIR, asset_path)


@app.post("/api/analyze")
def analyze() -> Any:
    model_key = request.form.get("model", "").strip().lower()
    file = request.files.get("image")

    if model_key not in DETECT_MODEL_PATHS:
        return jsonify({"error": "지원하지 않는 모델입니다. yolo26n 또는 yolo26s를 선택하세요."}), 400

    if file is None or not file.filename:
        return jsonify({"error": "이미지 파일이 필요합니다."}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "이미지를 읽을 수 없습니다."}), 400

    try:
        detect_output = _run_detection(image, model_key)
        roast_output = _run_roast_classification(image)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": f"추론 중 오류가 발생했습니다: {exc}"}), 500

    return jsonify(
        {
            "model": model_key,
            "device": DEVICE,
            "detection": detect_output,
            "roast": roast_output,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
