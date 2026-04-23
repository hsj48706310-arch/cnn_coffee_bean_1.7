# Results

> **Honest run**: 사용자 요청에 따라 두 task 모두 **클래스당 동일 규모** (≈1000장 풀)로 다운샘플링,  
> 두 task를 **완전히 분리된 single-task 모델**로 학습. (로스팅에 결점두를 섞지 않음)  
> Backbone: **EfficientNet-B0** (timm pretrained), AdamW + CosineLR, label smoothing 0.1,  
> 2-stage fine-tune (warmup 2ep backbone freeze → 6ep unfreeze), batch=32, lr=1e-3, seed=42.

---

## Task 1 — Roast Level (4-class)

데이터: Kaggle `gpiosenka/coffee-bean-dataset-resized-224-x-224`  
- 클래스: `green / light / medium / dark`  
- 샘플 수: 1,000장으로 stratified 다운샘플링 → train 700 / val 150 / test 150

| Metric | Val (best) | **Test** |
|---|---|---|
| Accuracy | — | **0.940** |
| Macro F1 | **0.9667** (E03) | **0.9400** |

Per-class test F1: green 0.987 · light 0.919 · medium 0.923 · dark 0.932

📊 [docs/confusion_matrix_roast.png](confusion_matrix_roast.png)

> **이전 ResNet18 풀데이터 (1600장) 실험에서 test F1 = 1.000** 이었으나,  
> 클래스당 샘플을 줄이자 0.94까지 떨어짐 → 1.0은 **데이터 포화**에 의한 천장 효과였음을 검증.  
> 0.94도 여전히 매우 높음 (배경 통일 + 색차 명확한 데이터셋 특성).

---

## Task 2 — Defect (17-class) ⭐ 메인 어트랙션

데이터: Kaggle `gpiosenka/coffee-bean-defects` (17-class, 약 6k장)  
- 클래스 (lowercase, 폴더명 오타 보존):  
  `broken, cut, dry cherry, fade, floater, full black, full sour,`  
  `fungus damange, husk, immature, parchment, partial black, partial sour,`  
  `severe insect damange, shell, slight insect damage, withered`
- 샘플 수: 979장으로 stratified 다운샘플링 → train 685 / val 147 / test 147

| Metric | Val (best) | **Test** |
|---|---|---|
| Accuracy | — | **0.782** |
| Macro F1 | **0.8003** (E07) | **0.7871** |

📊 [docs/confusion_matrix_defect.png](confusion_matrix_defect.png)

### Per-class test F1 (정렬)

| 강함 (≥0.85) | 중간 (0.70–0.85) | 약함 (<0.70) |
|---|---|---|
| dry cherry 1.000 | cut 0.778 | full sour 0.700 |
| full black 1.000 | fungus damange 0.750 | partial sour 0.706 |
| husk 1.000 | immature 0.714 | fade 0.429 |
| broken 0.941 | severe insect damange 0.714 | withered 0.375 |
| shell 0.941 | slight insect damage 0.800 |  |
| floater 0.857 | partial black 0.818 |  |
| parchment 0.857 |  |  |

### 분석
- **쉬운 클래스**: 색/형태가 뚜렷한 결점 (`dry cherry`, `full black`, `husk`) → 거의 완벽.
- **어려운 클래스**: `withered`(쪼그라듦), `fade`(약한 색변) → 정상 원두와 구분이 미묘. 추가 데이터 필요.
- **혼동 페어**: `partial sour ↔ full sour`, `slight insect damage ↔ severe insect damange` (동일 결함의 정도 차이 → 본질적으로 ordinal한 어려움).
- **2-stage fine-tune 효과**: warmup E02에서 F1=0.56 → unfreeze 후 E04에서 0.72로 점프 → E07 0.80 도달.

---

## 학습 곡선 요약

```
Roast :  E01 0.44  E02 0.78  E03 0.97 ← best (이후 saturate)
Defect:  E01 0.44  E02 0.56  E03 0.47  E04 0.72  E05 0.73  E06 0.75  E07 0.80 ← best  E08 0.78
```

- Roast는 3에폭만에 saturate (쉬운 task).
- Defect는 unfreeze 후 4에폭 더 점진 상승 (어려운 task에서 backbone fine-tune 효과 확인).

---

## Inference Speed (CPU, single image, ResNet18 baseline 측정값)

| Runtime | Latency | FPS |
|---|---|---|
| PyTorch CPU | 21.30 ms | 46.9 |
| **ONNX Runtime CPU** | **6.38 ms** | **156.8** |

→ ONNX 변환 시 **3.3× 가속**, 실시간(30 FPS) 요구 충분히 충족.

---

## Reproducibility

- seed=42 (numpy / torch / cuda / python random + cudnn deterministic)
- Roast 학습:  
  `python -m src.train --config configs/default.yaml --override train.epochs=8 train.warmup_epochs=2`
- Defect 학습:  
  `python -m src.train --config configs/defect.yaml --override train.epochs=8 train.warmup_epochs=2`
- 평가:  
  `python -m src.evaluate --config configs/default.yaml --ckpt checkpoints/best_roast.pth`  
  `python -m src.evaluate --config configs/defect.yaml --ckpt checkpoints/best_defect.pth`
- 두 모델 합본 metric:  
  `python -m scripts.dump_test_metrics` → `docs/test_metrics.json`
