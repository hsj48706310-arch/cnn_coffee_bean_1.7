"""Streamlit 데모: 이미지 업로드 -> 로스팅 단계 분류 + 확률 차트."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# 프로젝트 루트를 path에 추가 (streamlit run app/streamlit_app.py 실행 시)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, ROOT.as_posix())

import torch  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.dataset import get_transforms  # noqa: E402
from src.model import CoffeeClassifier  # noqa: E402
from src.evaluate import load_checkpoint  # noqa: E402

st.set_page_config(page_title="Coffee Bean AI", page_icon="☕", layout="centered")
st.title("☕ Coffee Bean Quality & Roasting AI")
st.caption("Kaggle 데이터로 학습한 CNN. 이미지를 업로드하면 로스팅 단계를 예측합니다.")

CFG_PATH = st.sidebar.text_input("config", "configs/default.yaml")
CKPT_PATH = st.sidebar.text_input("checkpoint (비우면 자동)", "")


@st.cache_resource
def load_model(cfg_path: str, ckpt_path: str):
    cfg = load_config(cfg_path)
    task, classes = cfg["task"], cfg["classes"]
    c2i = {c: i for i, c in enumerate(classes)}
    device = "cpu"
    ckpt_path = ckpt_path or f"{cfg['paths']['ckpt_dir']}/best_{task}.pth"
    ckpt_full = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt_full["model_state_dict"] if isinstance(ckpt_full, dict) and \
            "model_state_dict" in ckpt_full else ckpt_full
    ckpt_cfg = ckpt_full.get("config") if isinstance(ckpt_full, dict) else None
    backbone = (ckpt_cfg or cfg)["model"]["name"]
    model = CoffeeClassifier(
        backbone=backbone, n_classes=len(classes), pretrained=False,
        dropout=cfg["model"]["dropout"], hidden=cfg["model"]["hidden"],
    )
    model.load_state_dict(state); model.eval()
    tf = get_transforms(task=task, train=False, img_size=cfg["data"]["img_size"])
    return model, tf, classes, c2i, task


try:
    model, tf, classes, c2i, task = load_model(CFG_PATH, CKPT_PATH)
    st.success(f"모델 로딩 완료 — task={task}, classes={classes}")
except Exception as e:
    st.error(f"모델 로딩 실패: {e}\n먼저 학습을 실행해 checkpoint를 만들어 주세요.")
    st.stop()

up = st.file_uploader("원두 이미지 업로드", type=["jpg", "jpeg", "png"],
                      accept_multiple_files=True)

if up:
    rows = []
    for f in up:
        img = Image.open(f).convert("RGB")
        x = tf(image=np.array(img))["image"].unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).numpy()[0]
        top = int(probs.argmax())

        st.subheader(f.name)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(img, use_column_width=True)
        with c2:
            st.metric("예측", classes[top], f"{probs[top]*100:.1f}%")
            st.bar_chart(pd.DataFrame({"prob": probs}, index=classes))

        rows.append({"file": f.name, "pred": classes[top],
                     **{c: float(p) for c, p in zip(classes, probs)}})

    if len(rows) > 1:
        df = pd.DataFrame(rows)
        st.download_button("CSV 리포트 다운로드",
                           df.to_csv(index=False).encode("utf-8"),
                           file_name="coffee_predictions.csv",
                           mime="text/csv")
