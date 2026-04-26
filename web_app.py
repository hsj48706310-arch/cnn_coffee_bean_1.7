from __future__ import annotations

from pathlib import Path

import cv2
import gradio as gr
from ultralytics import YOLO


DETECTION_WEIGHTS = Path("artifacts") / "train" / "coffee_yolo26n_final" / "weights" / "best.pt"
CLASSIFICATION_WEIGHTS = Path("artifacts") / "cls" / "roast_cls_v8n_cpu" / "weights" / "best.pt"

CUSTOM_CSS = """
:root {
    --coffee-bg: #f5eee6;
    --coffee-card: #fffaf5;
    --coffee-border: #c9a37c;
    --coffee-text: #3d2a1f;
    --coffee-accent: #7a4e2d;
    --coffee-accent-2: #9a6a42;
}

body, .gradio-container {
    background: radial-gradient(circle at 0% 0%, #fdf3e6 0%, #f5eee6 45%, #efe2d4 100%);
    color: var(--coffee-text);
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container p,
.gradio-container label,
.gradio-container span,
.gradio-container div {
    color: var(--coffee-text);
}

.gradio-container .block,
.gradio-container .gr-box,
.gradio-container .gr-form,
.gradio-container .gr-panel {
    background: var(--coffee-card);
    border: 1px solid var(--coffee-border);
    border-radius: 12px;
}

.gradio-container button {
    background: linear-gradient(180deg, var(--coffee-accent-2) 0%, var(--coffee-accent) 100%);
    border: 1px solid #5b3922;
    color: #fff9f3;
}

.gradio-container button:hover {
    filter: brightness(1.06);
}
"""


def _load_model(weights_path: str | Path) -> YOLO:
    path = Path(weights_path)
    if not path.exists():
        raise gr.Error(f"모델 가중치 파일을 찾을 수 없습니다: {path}")
    return YOLO(str(path))


def run_detection(image, conf: float, iou: float, detection_weights_path: str):
    if image is None:
        raise gr.Error("먼저 이미지를 업로드해 주세요.")

    model = _load_model(detection_weights_path)
    result = model.predict(source=image, conf=conf, iou=iou, verbose=False)[0]

    plotted = result.plot()
    plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

    counts: dict[str, int] = {}
    if result.boxes is not None and result.boxes.cls is not None:
        for cls_id in result.boxes.cls.tolist():
            class_name = result.names.get(int(cls_id), str(int(cls_id)))
            counts[class_name] = counts.get(class_name, 0) + 1

    lines = ["검출 결과:"]
    if counts:
        for name, cnt in sorted(counts.items()):
            lines.append(f"- {name}: {cnt}")
    else:
        lines.append("- 검출 없음")

    return plotted_rgb, "\n".join(lines)


def run_roast_classification(image, classification_weights_path: str):
    if image is None:
        raise gr.Error("먼저 이미지를 업로드해 주세요.")

    model = _load_model(classification_weights_path)
    result = model.predict(source=image, verbose=False)[0]

    probs = result.probs
    if probs is None:
        raise gr.Error("분류 확률값을 가져오지 못했습니다.")

    top_idx = int(probs.top1)
    top_conf = float(probs.top1conf)
    class_name = result.names.get(top_idx, str(top_idx))

    confidence_table: dict[str, float] = {}
    if probs.data is not None:
        for idx, score in enumerate(probs.data.tolist()):
            confidence_table[result.names.get(idx, str(idx))] = float(score)

    summary = (
        f"예측 로스팅 클래스: {class_name}\n"
        f"신뢰도: {top_conf:.4f}"
    )
    return summary, confidence_table


def build_app() -> gr.Blocks:
    with gr.Blocks(title="커피 품질 통합 웹 UI", css=CUSTOM_CSS) as app:
        gr.Markdown("# 커피 원두 품질 검사 웹 UI")
        gr.Markdown(
            "결점두 검출과 로스팅 정도 판별을 하나의 앱에서 실행합니다. "
            "이미지를 올린 뒤 탭에서 기능을 선택해 주세요."
        )

        with gr.Tabs():
            with gr.Tab("결점두 검출"):
                det_weights = gr.Textbox(
                    value=str(DETECTION_WEIGHTS),
                    label="검출 모델 경로",
                    info="best.pt 경로를 직접 입력할 수 있습니다.",
                )
                det_image = gr.Image(type="numpy", label="입력 이미지")
                with gr.Row():
                    det_conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="신뢰도 임계값")
                    det_iou = gr.Slider(0.05, 0.95, value=0.45, step=0.05, label="IoU 임계값")
                det_button = gr.Button("결점두 검출 실행")
                det_output_img = gr.Image(type="numpy", label="검출 결과 이미지")
                det_output_text = gr.Textbox(label="요약", lines=8)

                det_button.click(
                    fn=run_detection,
                    inputs=[det_image, det_conf, det_iou, det_weights],
                    outputs=[det_output_img, det_output_text],
                )

            with gr.Tab("로스팅 정도 판별"):
                cls_weights = gr.Textbox(
                    value=str(CLASSIFICATION_WEIGHTS),
                    label="분류 모델 경로",
                    info="best.pt 경로를 직접 입력할 수 있습니다.",
                )
                cls_image = gr.Image(type="numpy", label="입력 이미지")
                cls_button = gr.Button("로스팅 판별 실행")
                cls_summary = gr.Textbox(label="예측 결과", lines=3)
                cls_scores = gr.Label(label="클래스별 확률")

                cls_button.click(
                    fn=run_roast_classification,
                    inputs=[cls_image, cls_weights],
                    outputs=[cls_summary, cls_scores],
                )

        gr.Markdown(
            "기본 모델 경로:\n"
            f"- 결점두 검출: {DETECTION_WEIGHTS}\n"
            f"- 로스팅 분류: {CLASSIFICATION_WEIGHTS}"
        )

    return app


if __name__ == "__main__":
    demo = build_app()
    demo.launch()