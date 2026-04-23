 # 설명 파일 위치 C:\Projects\mini_project\docs\WORKLOG.md  |  C:\Projects\mini_project\PROJECT_PLAN.txt 먼저 보시고 보시면 좋습니다
 



# Coffee Bean Quality & Roasting AI (cnn_coffee_bean_1.7)

원두 이미지를 입력받아 **로스팅 단계** 및 **결점두 여부**를 분류하는 CNN 기반
경량 솔루션. (Kaggle 공개 데이터셋만 사용 / Single-task MVP)

상세 설계는 [PROJECT_PLAN.txt](PROJECT_PLAN.txt) 참고.

## Quick Start

```powershell
# 1. 가상환경
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2. Kaggle API 토큰 준비 (%USERPROFILE%\.kaggle\kaggle.json)

# 3. 데이터 다운로드
python scripts/download_kaggle.py

# 4. 학습/평가/데모
# Roast (4-class)
python -m src.train    --config configs/default.yaml --override train.epochs=8 train.warmup_epochs=2
python -m src.evaluate --config configs/default.yaml --ckpt checkpoints/best_roast.pth
# Defect (17-class)
python -m src.train    --config configs/defect.yaml  --override train.epochs=8 train.warmup_epochs=2
python -m src.evaluate --config configs/defect.yaml  --ckpt checkpoints/best_defect.pth
streamlit run app/streamlit_app.py
```

## 구조

```
src/        모델/데이터/학습 코드
scripts/    Kaggle 다운로드, 스플릿 생성
configs/    YAML 하이퍼파라미터
data/       raw / annotations / splits  (raw는 git 제외)
app/        Streamlit 데모
notebooks/  EDA, 에러 분석
```

## 결과 요약 (정직한 다운샘플 실험)

> 각 task **별도 모델**, 클래스당 동일 규모(≈1000장 풀)로 다운샘플, EfficientNet-B0,  
> AdamW + CosineLR, 2-stage fine-tune (warmup 2ep → unfreeze 6ep), seed=42.

| Task | Classes | n (train/val/test) | Val F1 (best) | **Test acc** | **Test macro F1** |
|---|---|---|---|---|---|
| Roast  | 4  | 700 / 150 / 150 | 0.967 (E03) | **0.940** | **0.940** |
| Defect | 17 | 685 / 147 / 147 | 0.800 (E07) | **0.782** | **0.787** |

- Roast 1.0 → 0.94: 풀데이터(1600장)에서의 **천장 효과**가 다운샘플로 사라짐을 확인.
- Defect 17-class에서 0.80 도달: 2-stage fine-tune이 어려운 task에서 효과적임을 입증.
- 상세 분석/per-class F1/혼동 페어: [docs/results.md](docs/results.md)
- Confusion matrices: [roast](docs/confusion_matrix_roast.png) · [defect](docs/confusion_matrix_defect.png)

## 브랜치
- `main`  안정 버전
- `feat2` 본 작업 브랜치
