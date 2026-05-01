cnn_coffee_bean_1.7

학습 환경 설정 (YOLO26n + Roboflow export)

1) 데이터셋 zip 파일을 아래 경로에 압축 해제하세요.

datasets/Coffee Defect.v1-coffee-bean_yolo26n.yolo26

2) 프로젝트 폴더에서 의존성을 동기화하세요.

uv sync --active

3) 학습을 실행하세요.

uv run --active main.py

자주 쓰는 옵션:

uv run --active main.py --epochs 200 --batch 8 --imgsz 640
uv run --active main.py --device cpu

설치된 Ultralytics 빌드에 yolo26n 가중치가 없으면,
아래처럼 로컬 체크포인트 경로를 지정하세요.

--model path/to/yolo26n.pt

다음 경고가 보일 수 있습니다.

warning: `VIRTUAL_ENV=...` does not match the project environment path `.venv`

이 경우 위 예시처럼 같은 폴더에서 --active 옵션으로 실행하거나,
기존에 활성화된 다른 가상환경을 먼저 비활성화하세요.

출력 폴더 구조

artifacts/train  : 학습 실행 결과 및 가중치
artifacts/predict: 예측 결과 이미지/파일
artifacts/export : 내보낸 모델 파일

빠른 실행 스크립트

PowerShell -ExecutionPolicy Bypass -File scripts/train_gpu.ps1
PowerShell -ExecutionPolicy Bypass -File scripts/train_cpu.ps1

로스팅 분류기 학습 (EfficientNetV2-S)

PowerShell -ExecutionPolicy Bypass -File scripts/train_roast_gpu.ps1
PowerShell -ExecutionPolicy Bypass -File scripts/train_roast_cpu.ps1

권장 동시 학습 구성 (RTX 4060 Laptop GPU 1장 기준)

1) 결점두 검출(YOLO26n)은 GPU에서 배치를 낮춰 VRAM 스파이크를 줄여 실행합니다.
2) 로스팅 분류는 CPU에서 실행하거나, 둘 다 GPU를 쓸 경우 배치를 더 낮춰 실행합니다.

동시 실행 예시:

PowerShell -ExecutionPolicy Bypass -File scripts/train_gpu.ps1
PowerShell -ExecutionPolicy Bypass -File scripts/train_roast_cpu.ps1

로스팅 분류 데이터셋 생성 (crop 생성)

YOLO 주석을 기반으로 로스팅 단계 라벨
(`roasted-beans`, `under_roast`)용 분류 데이터셋을 별도로 생성합니다.

c:/miniproject1.7/.venv/Scripts/python.exe scripts/build_roast_cls_dataset.py --overwrite

빠른 점검 (작은 서브셋):

c:/miniproject1.7/.venv/Scripts/python.exe scripts/build_roast_cls_dataset.py --overwrite --max-label-files-per-split 1000

대표 샘플링 (빠른 검증 권장):

c:/miniproject1.7/.venv/Scripts/python.exe scripts/build_roast_cls_dataset.py --overwrite --max-label-files-per-split 1000 --shuffle-label-files

출력:

artifacts/roast_cls_dataset/train/<class_name>
artifacts/roast_cls_dataset/val/<class_name>
artifacts/roast_cls_dataset/test/<class_name>
artifacts/roast_cls_dataset/manifest.csv