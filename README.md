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
python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml
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

## 브랜치
- `main`  안정 버전
- `feat2` 본 작업 브랜치
