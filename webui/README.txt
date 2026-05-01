Web UI usage (backend connected)

1) Start the server from project root:
   python webui/server.py

2) Open this URL in browser:
   http://127.0.0.1:8000

3) First screen shows model selection buttons:
   - 모바일용 -> YOLO26n
   - 웹용 -> YOLO26s

4) Upload an image and click 분석 실행.
   The app runs both tasks in one request:
   - Defect bean detection (YOLO)
   - Roasting stage classification (EfficientNetV2-S)

Model paths used by default:
- YOLO26n: artifacts/train/coffee_yolo26n_e30_img832/weights/best.pt
- YOLO26s: artifacts/train/coffee_yolo26s_e5best_plus15_gpu/weights/best.pt
- Roast cls: artifacts/roast_cls_train_e5/best.pt
