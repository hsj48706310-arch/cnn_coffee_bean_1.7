"""실시간 웹캠 데모 (no save). 'q' 키로 종료."""
from __future__ import annotations
import argparse
import cv2
import numpy as np
import torch

from src.utils.config import load_config
from src.dataset import get_transforms
from src.model import CoffeeClassifier
from src.evaluate import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task, classes = cfg["task"], cfg["classes"]
    c2i = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in c2i.items()}

    ckpt_path = args.ckpt or f"{cfg['paths']['ckpt_dir']}/best_{task}.pth"
    _, state = load_checkpoint(ckpt_path, device)

    model = CoffeeClassifier(
        backbone=cfg["model"]["name"], n_classes=len(classes), pretrained=False,
        dropout=cfg["model"]["dropout"], hidden=cfg["model"]["hidden"],
    ).to(device)
    model.load_state_dict(state); model.eval()

    tf = get_transforms(task=task, train=False, img_size=cfg["data"]["img_size"])
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.cam}")

    print("[i] press 'q' to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = tf(image=rgb)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
        top = int(probs.argmax())
        text = f"{task}: {idx_to_class[top]}  p={probs[top]:.2f}"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(f"CoffeeBean ({task}) - q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
