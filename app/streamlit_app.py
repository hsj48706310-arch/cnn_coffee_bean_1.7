"""Streamlit demo: roast / defect classification toggle."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, ROOT.as_posix())

import torch  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.dataset import get_transforms  # noqa: E402
from src.model import CoffeeClassifier  # noqa: E402

st.set_page_config(page_title="Coffee Bean AI", page_icon="coffee", layout="centered")
st.title("Coffee Bean Quality and Roasting AI")
st.caption("EfficientNet-B0 dual model demo (roast 4-cls / defect 17-cls).")

TASKS = {
    "Roast (4-class)": {
        "config": "configs/default.yaml",
        "ckpt": "checkpoints/best_roast.pth",
        "metric": "Test acc 0.940 / macro F1 0.940",
    },
    "Defect (17-class)": {
        "config": "configs/defect.yaml",
        "ckpt": "checkpoints/best_defect.pth",
        "metric": "Test acc 0.782 / macro F1 0.787",
    },
}
choice = st.sidebar.radio("Model", list(TASKS.keys()))
sel = TASKS[choice]
st.sidebar.caption(sel["metric"])
st.sidebar.divider()
CFG_PATH = st.sidebar.text_input("config path", sel["config"])
CKPT_PATH = st.sidebar.text_input("checkpoint path", sel["ckpt"])


@st.cache_resource(show_spinner="Loading model...")
def load_model(cfg_path: str, ckpt_path: str):
    cfg = load_config(cfg_path)
    task, classes = cfg["task"], cfg["classes"]
    c2i = {c: i for i, c in enumerate(classes)}
    device = "cpu"
    ckpt_full = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt_full["model_state_dict"] if isinstance(ckpt_full, dict) and \
            "model_state_dict" in ckpt_full else ckpt_full
    ckpt_cfg = ckpt_full.get("config") if isinstance(ckpt_full, dict) else None
    backbone = (ckpt_cfg or cfg)["model"]["name"]
    model = CoffeeClassifier(
        backbone=backbone, n_classes=len(classes), pretrained=False,
        dropout=cfg["model"]["dropout"], hidden=cfg["model"]["hidden"],
    )
    model.load_state_dict(state)
    model.eval()
    tf = get_transforms(task=task, train=False, img_size=cfg["data"]["img_size"])
    return model, tf, classes, c2i, task, backbone


try:
    model, tf, classes, c2i, task, backbone = load_model(CFG_PATH, CKPT_PATH)
    st.success(f"Loaded: task={task}, backbone={backbone}, n_classes={len(classes)}")
except Exception as e:
    st.error(f"Load failed: {e}")
    st.stop()

up = st.file_uploader("Upload bean image(s)", type=["jpg", "jpeg", "png"],
                      accept_multiple_files=True)

TOPK = 3 if len(classes) <= 5 else 5

if up:
    rows = []
    for f in up:
        img = Image.open(f).convert("RGB")
        x = tf(image=np.array(img))["image"].unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).numpy()[0]
        order = probs.argsort()[::-1]
        top = int(order[0])

        st.subheader(f.name)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(img, use_container_width=True)
        with c2:
            st.metric("Prediction", classes[top], f"{probs[top]*100:.1f}%")
            topk_df = pd.DataFrame({
                "class": [classes[i] for i in order[:TOPK]],
                "prob":  [float(probs[i]) for i in order[:TOPK]],
            })
            st.dataframe(topk_df, hide_index=True, use_container_width=True)

        if len(classes) > 8:
            chart_df = pd.DataFrame({"prob": probs}, index=classes).sort_values("prob")
            st.bar_chart(chart_df, horizontal=True, height=400)
        else:
            st.bar_chart(pd.DataFrame({"prob": probs}, index=classes))

        rows.append({"file": f.name, "task": task, "pred": classes[top],
                     **{c: float(p) for c, p in zip(classes, probs)}})

    if len(rows) > 1:
        df = pd.DataFrame(rows)
        st.download_button("Download CSV report",
                           df.to_csv(index=False).encode("utf-8"),
                           file_name=f"coffee_predictions_{task}.csv",
                           mime="text/csv")
else:
    st.info("Pick a model in the sidebar and upload images to start.")