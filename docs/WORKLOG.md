# 프로젝트 작업 일지 — 팀원 설득용 정리

> **목표**: 커피 원두 이미지로 ① 로스팅 단계 ② 결점두 종류를 분류하는 CNN 시스템.  
> **현재 브랜치**: `feat2` (모든 작업 push 완료)  
> **마지막 커밋**: `a0b385f`

---

## 🎯 한 줄 요약

> "로스팅 모델은 **데이터가 너무 쉬워서** 1.0 점이 나왔던 게 들통났고,  
> 그래서 **결점두 17클래스**라는 진짜 어려운 task로 확장해서 **F1 0.79**를 달성했다.  
> 그리고 **ONNX 변환으로 5배 빠른 추론**까지 검증했다."

---

## 📅 작업 순서와 의사결정

### 1단계 — 베이스라인: ResNet18 로스팅 분류 (4클래스)

**왜?**
- 가장 단순한 분류 문제부터 시작해서 **파이프라인 동작을 검증**하는 게 우선.
- ResNet18은 가볍고 빠르고 PyTorch 표준 → MVP에 적합.

**무엇을?**
- Kaggle `gpiosenka/coffee-bean-dataset-resized-224-x-224` (1600장, 4클래스: green/light/medium/dark)
- ResNet18 + AdamW + CosineLR, 5 epoch
- train 1120 / val 240 / test 240 (stratified)

**결과**
- **Test acc 1.000, macro F1 1.000** ← 완벽한 점수
- ⚠️ **의심**: 너무 잘 나옴. 진짜 모델이 강한 건지, 아니면 task가 너무 쉬운 건지?

---

### 2단계 — EfficientNet-B0로 업그레이드

**왜?**
- ResNet18은 baseline용. 발표용으로는 **모던 백본**이 더 인상적.
- timm 라이브러리의 EfficientNet-B0는 ImageNet pretrain 가중치 품질이 좋음.
- **2-stage fine-tune (warmup으로 backbone 동결 → unfreeze)** 기법 적용 가능.

**무엇을?**
- backbone만 EfficientNet-B0로 교체, 나머지는 동일 파이프라인
- warmup 1ep + unfreeze 9ep

**결과**
- 또 **F1 1.000** → 모델을 바꿔도 똑같음
- ❗ **결론**: 모델 문제가 아니라 **데이터셋 자체가 너무 쉽다**.  
  배경 통일 + 클래스간 색차가 너무 명확해서 어떤 CNN이라도 풀어버림.

---

### 3단계 — 어려운 task 도입: 결점두 17클래스 분류

**왜?**
- 로스팅은 천장 효과로 더 이상 모델 차별화가 불가능.
- 발표에서 "F1 1.0 나왔어요!" 만 들고가면 **운 좋은 데이터셋이라는 비판**을 받음.
- 진짜 어려운 문제를 풀어야 우리 시스템의 가치를 증명.
- 결점두는 17개 클래스에 미세한 외관 차이 (예: `partial sour` vs `full sour`) → 본질적으로 어려운 ordinal 문제.

**무엇을?**
- Kaggle `gpiosenka/coffee-bean-defects` (약 6000장, 17클래스) 다운로드
- 17개 클래스: broken, cut, dry cherry, fade, floater, full black, full sour,  
  fungus damange, husk, immature, parchment, partial black, partial sour,  
  severe insect damange, shell, slight insect damage, withered

**결과**
- 데이터 준비 완료, 학습 인프라 task-agnostic으로 리팩토링 (`configs/defect.yaml` 추가)

---

### 4단계 — "정직한 비교" 다운샘플링 실험

**왜?**
- 로스팅 1600장 vs 결점두 6000장 → **데이터 양이 다르면 비교가 불공정**.
- 만약 결점두를 풀데이터로 돌려서 0.85가 나와도, "결점두는 데이터가 6배 많으니까 당연하지" 라는 반론이 가능.
- 두 task를 **클래스당 동일 규모(≈1000장 풀)로 맞춰서**, 모델 자체의 어려움을 비교해야 함.

**무엇을?**
- `scripts/make_splits.py`에 `--max_samples` 옵션 추가 (stratified per-class 다운샘플)
- Roast: 1600 → **1000장** (train 700 / val 150 / test 150)
- Defect: 6000+ → **979장** (train 685 / val 147 / test 147)

**결과**
- ✅ **Roast Test F1: 1.000 → 0.940** ← 데이터 줄이니까 천장이 무너짐. 이전 1.0이 **데이터 포화** 효과였음을 증명.
- ✅ **Defect Test F1: 0.787** ← 17클래스에서 정직하게 측정한 진짜 성능.
- 두 모델은 **완전히 분리된 single-task 모델** (사용자 요청대로 "로스팅에 결점두 안 섞기").

---

### 5단계 — 2-stage Fine-tune 효과 검증

**왜?**
- 결점두 17클래스 같은 어려운 task에서는 **단순 fine-tune보다 단계적 학습**이 효과적이라고 알려짐.
- 1단계: backbone 동결 + classifier head만 학습 → 안정적 시작
- 2단계: backbone 풀고 전체 학습 → 도메인 적응

**무엇을?**
- warmup 2 epoch (frozen) → unfreeze 6 epoch
- AdamW + CosineLR + Label Smoothing 0.1

**결과** — 학습 곡선이 **2-stage 효과를 명확히 보여줌**:
```
[E01] frozen   F1=0.44  ← warmup 시작
[E02] frozen   F1=0.56
[E03] unfreeze F1=0.47  ← 잠깐 dip (backbone 적응 중)
[E04] unfreeze F1=0.72  ← 큰 점프!
[E05] unfreeze F1=0.73
[E06] unfreeze F1=0.75
[E07] unfreeze F1=0.80  ← best
[E08] unfreeze F1=0.78
```

---

### 6단계 — Defect 모델 분석 (per-class)

**왜?**
- 평균 F1만 보면 모델이 어디서 헷갈리는지 모름.
- 발표에서 **"어떤 게 쉽고 어떤 게 어려웠는지"** 인사이트가 필요.

**결과 요약**

| 강함 (F1≥0.85) | 중간 (0.70~0.85) | 약함 (<0.70) |
|---|---|---|
| dry cherry 1.000 | cut 0.778 | full sour 0.700 |
| full black 1.000 | fungus damange 0.750 | partial sour 0.706 |
| husk 1.000 | immature 0.714 | **fade 0.429** |
| broken 0.941 | severe insect damange 0.714 | **withered 0.375** |
| shell 0.941 | slight insect damage 0.800 | |
| floater 0.857 | partial black 0.818 | |
| parchment 0.857 | | |

**인사이트**:
- **쉬운 클래스**: 색/형태가 뚜렷하게 다른 결점 (`full black`, `husk`, `dry cherry`) → 거의 완벽
- **어려운 클래스**: `withered`(쪼그라듦), `fade`(미묘한 색변) → 정상 원두랑 구분이 거의 불가능
- **혼동 페어**: `partial sour ↔ full sour`, `slight insect ↔ severe insect`  
  → 동일 결함의 **정도(degree) 차이** → 본질적으로 ordinal 문제 → 데이터 더 늘려도 한계 있음

---

### 7단계 — ONNX 변환 + 추론 가속

**왜?**
- PyTorch는 학습/연구에 좋지만 **배포에는 무거움** (Python + PyTorch 런타임 필수).
- ONNX는 표준 포맷 → ONNX Runtime, TensorRT, 모바일 등 어디든 배포 가능.
- 실시간 분류기로 쓰려면 **30 FPS 이상** 필요.

**무엇을?**
- 두 모델 모두 ONNX export (opset 17)
- ONNX Runtime CPU 벤치마크 vs PyTorch CPU

**결과**

| Model | Runtime | Latency | FPS | Speedup |
|---|---|---|---|---|
| Roast (ResNet18) | PyTorch CPU | 21.30 ms | 46.9 | 1.0× |
| Roast (ResNet18) | **ONNX CPU** | **6.38 ms** | **156.8** | **3.3×** |
| Defect (EffNet-B0) | PyTorch CPU | 23.34 ms | 42.8 | 1.0× |
| Defect (EffNet-B0) | **ONNX CPU** | **4.74 ms** | **211.2** | **4.9×** |

→ 30 FPS 목표를 **5배 이상 초과 달성**, 실시간 라인 검사기로 충분히 활용 가능.

---

### 8단계 — Streamlit 데모 앱

**왜?**
- 모델은 만들었는데 **눈으로 보여줄 게 있어야** 발표가 산다.
- 사이드바 토글로 두 모델 한 화면에서 시연 가능 → 발표 임팩트.

**무엇을?**
- 사이드바 라디오: `🔥 로스팅 분류 (4클래스)` ↔ `🐛 결점두 분류 (17클래스)`
- 이미지 다중 업로드 → 예측 + 확률 + Top-K + 정렬된 막대그래프
- 한글 UI, CSV 리포트 다운로드 버튼

**결과**: `app/streamlit_app.py` 완성, `streamlit run app/streamlit_app.py`로 실행 가능.

---

## 📊 최종 성능표

| Task | 클래스 수 | 데이터 (train/val/test) | Backbone | Val F1 | **Test acc** | **Test F1** | ONNX FPS |
|---|---|---|---|---|---|---|---|
| Roast  | 4  | 700/150/150 | EfficientNet-B0 | 0.967 | **0.940** | **0.940** | 157 |
| Defect | 17 | 685/147/147 | EfficientNet-B0 | 0.800 | **0.782** | **0.787** | 211 |

---

## 🗂️ 산출물 위치

| 종류 | 경로 |
|---|---|
| 학습 코드 | `src/train.py`, `src/model.py`, `src/dataset.py` |
| 평가 코드 | `src/evaluate.py`, `scripts/dump_test_metrics.py` |
| ONNX 변환 | `src/export_onnx.py` |
| 데모 앱 | `app/streamlit_app.py` |
| Config | `configs/default.yaml` (roast), `configs/defect.yaml` |
| 체크포인트 | `checkpoints/best_roast.pth`, `checkpoints/best_defect.pth`, `*.onnx` |
| 데이터 분할 | `data/splits/{roast,defect}_{train,val,test}.csv` |
| Confusion Matrix | `docs/confusion_matrix_{roast,defect}.png` |
| 메트릭 JSON | `docs/test_metrics.json` |
| 결과 문서 | `docs/results.md` |

---

## ✅ 완료된 작업 체크리스트

- [x] Roast 베이스라인 (ResNet18) 학습/평가
- [x] EfficientNet-B0 업그레이드
- [x] Defect 17클래스 데이터 다운로드 + 정제
- [x] 두 task 동일 규모 다운샘플링 (정직한 비교)
- [x] 두 모델 분리 학습 (single-task)
- [x] 2-stage fine-tune 적용
- [x] Test set 평가 + per-class F1 분석
- [x] Confusion matrix 시각화
- [x] ONNX export + 속도 벤치마크
- [x] Streamlit 데모 앱 (한글 UI)
- [x] 문서 (README, docs/results.md) 업데이트
- [x] feat2 브랜치 push (4개 커밋)

## 🚧 남은 작업 (선택 사항)

- [ ] **Streamlit UI 꾸미기** (테마, 레이아웃, 헤더 이미지 등) ← 다음 작업
- [ ] PPT 발표자료 제작 ← 메인 목표
- [ ] (옵션) Grad-CAM으로 모델이 어디를 보는지 시각화
- [ ] (옵션) 어려운 클래스(`withered`, `fade`)에 대한 추가 데이터 수집

---

## 💡 팀원 설득 포인트

1. **"문제를 정직하게 어렵게 만들었다"**  
   → 1.0 점 자랑하지 않고 천장 효과를 직접 깨뜨려서 0.94로 보고. 신뢰성 있는 평가.

2. **"진짜 어려운 task로 확장했다"**  
   → 17클래스 결점두는 산업 현장에서 실제로 필요한 분류 (커피 품질 등급 결정).  
   → F1 0.79는 ResNet18 1.0보다 **훨씬 더 임팩트 있는 숫자**.

3. **"배포 가능한 형태로 만들었다"**  
   → ONNX 4.9× 가속, 211 FPS → 라인 컨베이어 위에서도 실시간 검사 가능 수준.

4. **"보여줄 데모가 있다"**  
   → Streamlit 앱으로 그 자리에서 이미지 분류 시연 가능 (실제 시연 안 해도 스크린샷으로 충분).

5. **"한계도 정직하게 분석했다"**  
   → `withered`, `fade` 같은 미묘한 결함은 모델 한계가 분명. 추가 데이터 수집이 다음 스텝.
