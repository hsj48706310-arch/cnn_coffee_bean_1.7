"""Bean and Brew AI - cafe themed Streamlit demo."""
from __future__ import annotations
import json
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

st.set_page_config(
    page_title="Bean & Brew AI",
    page_icon="\u2615",
    layout="wide",
    initial_sidebar_state="expanded",
)

CAFE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Nanum+Myeongjo:wght@400;700;800&family=Caveat:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Nanum Myeongjo', 'Playfair Display', serif;
}

.cafe-header {
    background: linear-gradient(135deg, #3E2723 0%, #6F4E37 50%, #A0785A 100%);
    padding: 2.5rem 2rem;
    border-radius: 18px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 24px rgba(62, 39, 35, 0.25);
    text-align: center;
    color: #FAF7F2;
}
.cafe-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    margin: 0;
    letter-spacing: 2px;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
}
.cafe-header .subtitle {
    font-family: 'Caveat', cursive;
    font-size: 1.6rem;
    margin-top: 0.5rem;
    color: #F0E6D8;
}
.cafe-header .tagline {
    font-size: 0.95rem;
    margin-top: 0.8rem;
    opacity: 0.85;
    letter-spacing: 1px;
}

.menu-card {
    background: #FFFEFB;
    border: 2px solid #C8A27A;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin: 0.6rem 0;
    box-shadow: 0 4px 12px rgba(111, 78, 55, 0.12);
}
.menu-card h3 {
    color: #3E2723;
    font-family: 'Playfair Display', serif;
    border-bottom: 2px dashed #C8A27A;
    padding-bottom: 0.6rem;
    margin-bottom: 1rem;
}
.menu-card .price {
    float: right;
    color: #6F4E37;
    font-weight: 700;
}

.kpi-card {
    background: linear-gradient(145deg, #FFFEFB, #F0E6D8);
    border-left: 5px solid #6F4E37;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 3px 10px rgba(111, 78, 55, 0.1);
    height: 100%;
}
.kpi-card .label {
    font-size: 0.85rem;
    color: #8D6E63;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.kpi-card .value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #3E2723;
    font-family: 'Playfair Display', serif;
    margin: 0.3rem 0;
}
.kpi-card .delta {
    font-size: 0.9rem;
    color: #6F4E37;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F0E6D8 0%, #E8D9C4 100%);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #3E2723;
    font-family: 'Playfair Display', serif;
}

[data-testid="stTabs"] button {
    font-family: 'Nanum Myeongjo', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #6F4E37;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #3E2723;
    border-bottom: 3px solid #6F4E37 !important;
}

.stButton > button, .stDownloadButton > button {
    background: #6F4E37;
    color: #FAF7F2;
    border: none;
    border-radius: 25px;
    padding: 0.5rem 1.4rem;
    font-family: 'Nanum Myeongjo', serif;
    font-weight: 700;
    transition: all 0.2s;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #3E2723;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(62, 39, 35, 0.3);
}

[data-testid="stAlert"] {
    border-radius: 12px;
    border-left-width: 5px;
}

.cafe-footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    color: #8D6E63;
    font-family: 'Caveat', cursive;
    font-size: 1.2rem;
    border-top: 1px dashed #C8A27A;
    margin-top: 3rem;
}

.chapter-divider {
    text-align: center;
    margin: 2rem 0 1rem 0;
    color: #6F4E37;
    font-family: 'Caveat', cursive;
    font-size: 1.8rem;
}
.chapter-divider::before, .chapter-divider::after {
    content: " \\2615 ";
    margin: 0 1rem;
}
</style>
"""
st.markdown(CAFE_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="cafe-header">
        <h1>Bean &amp; Brew AI</h1>
        <div class="subtitle">~ Roastery x Quality Lab ~</div>
        <div class="tagline">EST. 2026 | POWERED BY EFFICIENTNET-B0 | CRAFTED WITH LOVE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

TASKS = {
    "\U0001F525 \ub85c\uc2a4\ud305 \ub2e8\uacc4 \ubd84\ub958": {
        "config": "configs/default.yaml",
        "ckpt": "checkpoints/best_roast.pth",
        "menu_name": "Roast Level Espresso",
        "tagline": "\uc6d0\ub450\uc758 \uc775\ud798 \uc815\ub3c4\ub97c \ud310\ubcc4\ud569\ub2c8\ub2e4",
        "price": "4 CLASSES",
        "test_acc": 0.940,
        "test_f1": 0.940,
        "val_f1": 0.967,
    },
    "\U0001F41B \uacb0\uc810\ub450 \uc885\ub958 \ubd84\ub958": {
        "config": "configs/defect.yaml",
        "ckpt": "checkpoints/best_defect.pth",
        "menu_name": "Defect Discovery Latte",
        "tagline": "17\uac00\uc9c0 \uacb0\uc810\ub450\ub97c \uc815\ubc00\ud558\uac8c \uc2dd\ubcc4\ud569\ub2c8\ub2e4",
        "price": "17 CLASSES",
        "test_acc": 0.782,
        "test_f1": 0.787,
        "val_f1": 0.800,
    },
}

with st.sidebar:
    st.markdown("## \u2615 \uc624\ub298\uc758 \uba54\ub274")
    st.markdown("*Today's Menu*")
    st.divider()

    choice = st.radio(
        "\uc6d0\ud558\uc2dc\ub294 \ubd84\uc11d\uc744 \uace8\ub77c\uc8fc\uc138\uc694",
        list(TASKS.keys()),
        label_visibility="collapsed",
    )
    sel = TASKS[choice]

    st.markdown(
        f"""
        <div class="menu-card" style="margin-top: 1rem;">
            <h3>{sel['menu_name']} <span class="price">{sel['price']}</span></h3>
            <p style="color:#6F4E37; font-style:italic; margin:0;">
                {sel['tagline']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### \U0001F4CA \ubaa8\ub378 \uce74\ub4dc")
    st.markdown(f"- **\ud14c\uc2a4\ud2b8 \uc815\ud655\ub3c4**: `{sel['test_acc']:.1%}`")
    st.markdown(f"- **\ud14c\uc2a4\ud2b8 Macro F1**: `{sel['test_f1']:.3f}`")
    st.markdown(f"- **\uac80\uc99d Macro F1**: `{sel['val_f1']:.3f}`")

    st.divider()
    with st.expander("\u2699\ufe0f \uace0\uae09 \uc124\uc815"):
        CFG_PATH = st.text_input("config \uacbd\ub85c", sel["config"])
        CKPT_PATH = st.text_input("checkpoint \uacbd\ub85c", sel["ckpt"])

    st.markdown(
        """
        <div style="text-align:center; margin-top:2rem; color:#6F4E37; font-family:'Caveat', cursive; font-size:1.1rem;">
            "Life is too short<br>for bad coffee"<br>- Bean &amp; Brew
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="\u2615 \ubaa8\ub378\uc744 \ub04c\uc774\ub294 \uc911...")
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


@st.cache_data
def load_metrics():
    p = ROOT / "docs" / "test_metrics.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


METRICS = load_metrics()

try:
    model, tf, classes, c2i, task, backbone = load_model(CFG_PATH, CKPT_PATH)
    model_loaded = True
    load_err = ""
except Exception as e:
    model_loaded = False
    load_err = str(e)
    task = "roast"
    classes = []


tab_home, tab_predict, tab_perf, tab_about = st.tabs([
    "\U0001F3E0 \ud648 (Welcome)",
    "\U0001F50D \uc6d0\ub450 \ubd84\uc11d (Tasting)",
    "\U0001F4CA \uc131\ub2a5 \ub9ac\ud3ec\ud2b8 (Cupping Notes)",
    "\U0001F4D6 \ubaa8\ub378 \uc774\uc57c\uae30 (Our Story)",
])

with tab_home:
    st.markdown('<div class="chapter-divider">Welcome to Our Roastery</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("\U0001F525 \ub85c\uc2a4\ud305 \uc815\ud655\ub3c4", "94.0%", "4-class | F1 0.940"),
        ("\U0001F41B \uacb0\uc810\ub450 \uc815\ud655\ub3c4", "78.2%", "17-class | F1 0.787"),
        ("\u26A1 \ucc98\ub9ac \uc18d\ub3c4", "211 FPS", "ONNX | CPU"),
        ("\u2615 \ud559\uc2b5 \ub370\uc774\ud130", "1,950+", "\uace0\ud488\uc9c8 \uc6d0\ub450"),
    ]
    for col, (label, value, delta) in zip([col1, col2, col3, col4], cards):
        col.markdown(
            f"""
            <div class="kpi-card">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
                <div class="delta">{delta}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="chapter-divider">Our Signature Beans</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            <div class="menu-card">
                <h3>\U0001F525 Roast Level Espresso <span class="price">W 4 CLASSES</span></h3>
                <p style="color:#5D4037;">
                    \uc6d0\ub450\uc758 \ub85c\uc2a4\ud305 \ub2e8\uacc4\ub97c
                    <b>green / light / medium / dark</b> 4\ub2e8\uacc4\ub85c \uc815\ubc00 \ubd84\ub958.<br>
                    <i>Notes - \uade0\ub4f1 \ub2e4\uc6b4\uc0d8\ud50c\ub9c1, \uc815\uc9c1\ud55c \ud3c9\uac00,
                    \ucc9c\uc7a5 \ud6a8\uacfc \uadf9\ubcf5</i>
                </p>
                <p style="color:#6F4E37; margin:0;">
                    \U0001F3C6 Macro F1 <b>0.940</b> | \ud14c\uc2a4\ud2b8 \uc815\ud655\ub3c4 <b>94.0%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """
            <div class="menu-card">
                <h3>\U0001F41B Defect Discovery Latte <span class="price">W 17 CLASSES</span></h3>
                <p style="color:#5D4037;">
                    SCA \ud45c\uc900 \uacb0\uc810\ub450 17\uc885\uc744 \uc790\ub3d9 \ud310\ubcc4.<br>
                    broken, fungus, husk, immature, partial sour, withered ...<br>
                    <i>Notes - 2-stage fine-tuning, label smoothing, cosine LR</i>
                </p>
                <p style="color:#6F4E37; margin:0;">
                    \U0001F3C6 Macro F1 <b>0.787</b> | \ud14c\uc2a4\ud2b8 \uc815\ud655\ub3c4 <b>78.2%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="chapter-divider">How to Order</div>', unsafe_allow_html=True)
    st.markdown(
        """
        1. \uc67c\ucabd **\uba54\ub274\ud310**\uc5d0\uc11c \uc6d0\ud558\ub294 \ubd84\uc11d\uc744 \uace0\ub974\uc138\uc694 \u2615
        2. **\U0001F50D \uc6d0\ub450 \ubd84\uc11d** \ud0ed\uc73c\ub85c \uc774\ub3d9\ud574\uc11c \uc6d0\ub450 \uc0ac\uc9c4\uc744 \uc5c5\ub85c\ub4dc\ud558\uc138\uc694 \U0001F4F8
        3. AI \ubc14\ub9ac\uc2a4\ud0c0\uac00 \uacb0\uacfc\ub97c \uc815\uc131\uaecf \ub0b4\ub824\ub4dc\ub9bd\ub2c8\ub2e4 \u2728
        4. \ub354 \uc790\uc138\ud55c \uacb0\uacfc\ub294 **\U0001F4CA \uc131\ub2a5 \ub9ac\ud3ec\ud2b8** \uc5d0\uc11c \ud655\uc778\ud558\uc138\uc694 \U0001F4C8
        """
    )

with tab_predict:
    st.markdown('<div class="chapter-divider">Tasting Bar - Bring Your Beans</div>',
                unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"\u274C \ubaa8\ub378 \ub85c\ub529 \uc2e4\ud328: {load_err}")
    else:
        st.success(
            f"\u2705 **{sel['menu_name']}** \ubaa8\ub378\uc774 \uc900\ube44\ub418\uc5c8\uc2b5\ub2c8\ub2e4  |  "
            f"task=`{task}` | backbone=`{backbone}` | \ud074\ub798\uc2a4 `{len(classes)}\uac1c`"
        )

        up = st.file_uploader(
            "\U0001F4F8 \uc6d0\ub450 \uc774\ubbf8\uc9c0\ub97c \uc5c5\ub85c\ub4dc\ud574\uc8fc\uc138\uc694 (\uc5ec\ub7ec \uc7a5 \uac00\ub2a5)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

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

                st.markdown(f"#### \u2615 {f.name}")
                c1, c2 = st.columns([1, 1.2])
                with c1:
                    st.image(img, use_container_width=True)
                with c2:
                    st.markdown(
                        f"""
                        <div class="kpi-card" style="margin-bottom:0.8rem;">
                            <div class="label">Top-1 \uc608\uce21 \uacb0\uacfc</div>
                            <div class="value" style="font-size:1.8rem;">{classes[top]}</div>
                            <div class="delta">\uc2e0\ub8b0\ub3c4 {probs[top]*100:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    topk_df = pd.DataFrame({
                        "\uc21c\uc704": [f"#{i+1}" for i in range(TOPK)],
                        "\ud074\ub798\uc2a4": [classes[i] for i in order[:TOPK]],
                        "\ud655\ub960": [f"{float(probs[i])*100:.1f}%" for i in order[:TOPK]],
                    })
                    st.dataframe(topk_df, hide_index=True, use_container_width=True)

                with st.expander("\U0001F4CA \uc804\uccb4 \ud074\ub798\uc2a4 \ud655\ub960 \ubd84\ud3ec \ubcf4\uae30"):
                    if len(classes) > 8:
                        chart_df = pd.DataFrame({"\ud655\ub960": probs}, index=classes).sort_values("\ud655\ub960")
                        st.bar_chart(chart_df, horizontal=True, height=400)
                    else:
                        st.bar_chart(pd.DataFrame({"\ud655\ub960": probs}, index=classes))

                st.divider()
                rows.append({"file": f.name, "task": task, "pred": classes[top],
                             **{c: float(p) for c, p in zip(classes, probs)}})

            if len(rows) > 1:
                df = pd.DataFrame(rows)
                st.download_button(
                    "\U0001F4E5 CSV \uc601\uc218\uc99d \ub2e4\uc6b4\ub85c\ub4dc",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name=f"bean_brew_receipt_{task}.csv",
                    mime="text/csv",
                )
        else:
            st.info(
                "\U0001F446 \uc704\ucabd\uc758 \uc5c5\ub85c\ub354\uc5d0 \uc6d0\ub450 \uc0ac\uc9c4\uc744 \uc62c\ub824\uc8fc\uc2dc\uba74 "
                "AI \ubc14\ub9ac\uc2a4\ud0c0\uac00 \ubd84\uc11d\uc744 \uc2dc\uc791\ud569\ub2c8\ub2e4.\n\n"
                "\ud301: \ud55c \ubc88\uc5d0 \uc5ec\ub7ec \uc7a5 \uc62c\ub9ac\uba74 \ud55c\uaebc\ubc88\uc5d0 \ubd84\uc11d\ud560 \uc218 \uc788\uc5b4\uc694!"
            )

with tab_perf:
    st.markdown('<div class="chapter-divider">Cupping Notes - \uc815\uc9c1\ud55c \ud3c9\uac00\ud45c</div>',
                unsafe_allow_html=True)

    task_key = "roast" if task == "roast" else "defect"
    m = METRICS.get(task_key, {})
    if not m:
        st.warning("test_metrics.json \ud30c\uc77c\uc744 \ucc3e\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        kpis = [
            ("\ud14c\uc2a4\ud2b8 \uc0d8\ud50c \uc218", f"{m['n_test']}"),
            ("\ud14c\uc2a4\ud2b8 \uc815\ud655\ub3c4", f"{m['test_acc']*100:.1f}%"),
            ("\ud14c\uc2a4\ud2b8 Macro F1", f"{m['test_macro_f1']:.3f}"),
            ("\uac80\uc99d Macro F1", f"{m['ckpt_val_macro_f1']:.3f}"),
        ]
        for col, (label, value) in zip([c1, c2, c3, c4], kpis):
            col.markdown(
                f"""
                <div class="kpi-card">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                    <div class="delta">@ epoch {m['ckpt_epoch']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### \U0001F36B \ud074\ub798\uc2a4\ubcc4 F1 \uc810\uc218")
        f1_df = pd.DataFrame({
            "\ud074\ub798\uc2a4": list(m["per_class_f1"].keys()),
            "F1": list(m["per_class_f1"].values()),
        }).sort_values("F1", ascending=True)
        st.bar_chart(f1_df.set_index("\ud074\ub798\uc2a4"), horizontal=True,
                     height=max(280, 28 * len(f1_df)))

        st.markdown("### \U0001F3A8 \ud63c\ub3d9 \ud589\ub82c (Confusion Matrix)")
        cm_path = ROOT / "docs" / f"confusion_matrix_{task_key}.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True,
                     caption=f"\ud14c\uc2a4\ud2b8\uc14b {m['n_test']}\uc7a5 \uae30\uc900 - \ub300\uac01\uc120\uc774 \uc815\ub2f5, \uadf8 \uc678\ub294 \uc624\ubd84\ub958")
        else:
            st.info("\ud63c\ub3d9 \ud589\ub82c \uc774\ubbf8\uc9c0\ub97c \ucc3e\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4.")

with tab_about:
    st.markdown('<div class="chapter-divider">Our Story - \ud55c \uc794\uc758 \ubaa8\ub378\uc774 \ub9cc\ub4e4\uc5b4\uc9c0\uae30\uae4c\uc9c0</div>',
                unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown(
            """
            ### \u2615 \ubc31\ubcf8 (Backbone)
            **EfficientNet-B0** (2019, Google Brain)
            - \ud30c\ub77c\ubbf8\ud130 5.3M | FLOPs 0.39G
            - ImageNet Top-1 **77.1%**
            - NAS \uae30\ubc18 \ud6a8\uc728 \ucd5c\uc801\ud654 \uc124\uacc4

            ### \U0001F36F \ud559\uc2b5 \ub808\uc2dc\ud53c
            - **Optimizer** AdamW (lr=3e-4, wd=1e-4)
            - **Scheduler** Cosine Annealing
            - **Loss** CrossEntropy + Label Smoothing 0.1
            - **Stage 1** Backbone freeze | 2 epochs (warmup)
            - **Stage 2** Full fine-tune | 6 epochs
            - **Augmentation** Albumentations
            """
        )
    with col_r:
        st.markdown(
            """
            ### \U0001F95B \ub370\uc774\ud130 (Beans)
            | \ubb38\uc81c | \ud074\ub798\uc2a4 | Train | Val | Test |
            |---|---|---|---|---|
            | \U0001F525 \ub85c\uc2a4\ud305 | 4 | 700 | 150 | 150 |
            | \U0001F41B \uacb0\uc810\ub450 | 17 | 685 | 147 | 147 |

            > **Stratified split** \uc73c\ub85c \ud074\ub798\uc2a4 \ube44\uc728 \uc720\uc9c0

            ### \u26A1 \ucd94\ub860 \ud658\uacbd (Brewing)
            | \ubaa8\ub378 | PyTorch | ONNX | \uac00\uc18d |
            |---|---|---|---|
            | \U0001F525 \ub85c\uc2a4\ud305 | 47 FPS | **157 FPS** | 3.3x |
            | \U0001F41B \uacb0\uc810\ub450 | 43 FPS | **211 FPS** | 4.9x |

            > \ubaa8\ub450 CPU \uae30\uc900 | 30 FPS \ubaa9\ud45c\ub97c 5\ubc30 \uc774\uc0c1 \ucd08\uacfc
            """
        )

    st.markdown('<div class="chapter-divider">Bean Selection Notes</div>', unsafe_allow_html=True)
    st.markdown(
        """
        \U0001F3F7\ufe0f **\uc65c EfficientNet-B0 \uc778\uac00?**
        ResNet18(11.7M / 1.8G / 69.8%), ResNet34(21.8M / 3.7G / 73.3%),
        ResNet50(25.6M / 4.1G / 76.1%) \ub300\ube44 \ud30c\ub77c\ubbf8\ud130 | \uc5f0\uc0b0\ub7c9 | \uc815\ud655\ub3c4 \uc138 \uc9c0\ud45c \ubaa8\ub450\uc5d0\uc11c
        \uc6b0\uc138\ud558\uac70\ub098 \ub3d9\ub4f1\ud55c EfficientNet-B0(5.3M / 0.39G / 77.1%) \ucc44\ud0dd.

        \U0001F3AF **\ud575\uc2ec \ubc1c\uacac**
        1\ub2e8\uacc4 ResNet18 \ubca0\uc774\uc2a4\ub77c\uc778\uc5d0\uc11c F1=1.000 \uc758 \ube44\ud604\uc2e4\uc801 \uc810\uc218\uac00 \ub098\uc654\uc73c\uba70,
        \uade0\ub4f1 \ub2e4\uc6b4\uc0d8\ud50c\ub9c1 \ud6c4 F1=0.940 \uc73c\ub85c \ubcf4\uc815. **\ucc9c\uc7a5 \ud6a8\uacfc(ceiling effect)** \ub97c \uc9c1\uc811 \ud655\uc778.

        \U0001F52C **17-class \ud55c\uacc4 \uc778\uc815**
        `withered` (F1 0.375), `fade` (F1 0.429) \ub294 \uc815\uc0c1 \uc6d0\ub450\uc640 \uc2dc\uac01\uc801 \uad6c\ubd84\uc774 \uc5b4\ub824\uc6b4 \uacb0\uc810\uc774\uba70,
        \ubcf8 \ubaa8\ub378\uc758 \ud55c\uacc4\ub97c \uba85\ud655\ud788 \ubcf4\uace0. \ucd94\uac00 \ub370\uc774\ud130 / Grad-CAM \ubd84\uc11d\uc774 \ud6c4\uc18d \uacfc\uc81c.
        """
    )

st.markdown(
    """
    <div class="cafe-footer">
        \u2615 Bean &amp; Brew AI | Brewed with PyTorch &amp; Streamlit | 2026 \u2615<br>
        <span style="font-size:0.95rem; color:#A0785A;">
            "We don't sell coffee, we sell experience."
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
