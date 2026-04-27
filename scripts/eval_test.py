#!/usr/bin/env python3
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from pathlib import Path
import numpy as np

# 설정
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224
BATCH_SIZE = 256
DEVICE = torch.device("cpu")

# 테스트셋 로드
test_root = Path('artifacts/roast_cls_dataset_sample/test').resolve()
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

test_ds = datasets.ImageFolder(test_root, transform=test_transform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"테스트셋: {len(test_ds)} 이미지")
print(f"클래스: {test_ds.classes}\n")

# 모델 로드
checkpoint = torch.load(Path('artifacts/roast_cls_train_e5/best.pt').resolve(), map_location=DEVICE)
weights = EfficientNet_V2_S_Weights.DEFAULT
model = efficientnet_v2_s(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(checkpoint['class_to_idx']))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

print(f"모델 로드 완료 (Epoch: {checkpoint['epoch']}, 학습 검증 정확도: {checkpoint['val_acc']:.4f})\n")

# 추론
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        
        # 진행 상황 출력 (매 10 배치마다)
        if (batch_idx + 1) % 10 == 0:
            progress = 100 * (batch_idx + 1) / len(test_loader)
            print(f"처리 중... [{batch_idx+1}/{len(test_loader)}] ({progress:.1f}%)", flush=True)

print("\n" + "=" * 70)
print("테스트셋 평가 결과")
print("=" * 70)

# 전체 정확도
accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f"\n전체 정확도: {100*accuracy:.2f}% ({np.sum(np.array(all_preds) == np.array(all_labels))}/{len(all_labels)})\n")

# 클래스별 정확도
print("클래스별 정확도:")
for class_idx, class_name in enumerate(test_ds.classes):
    mask = np.array(all_labels) == class_idx
    if np.sum(mask) > 0:
        correct = np.sum(np.array(all_preds)[mask] == class_idx)
        total = np.sum(mask)
        class_acc = correct / total
        print(f"  {class_name:15s}: {100*class_acc:.2f}% ({correct}/{total})")

# 혼동 행렬 (수동 계산)
print(f"\n혼동 행렬:")
cm = [[0, 0], [0, 0]]
for pred, label in zip(all_preds, all_labels):
    cm[label][pred] += 1

print(f"  Predicted -->")
print(f"  {test_ds.classes[0]:15s}: {cm[0]}")
print(f"  {test_ds.classes[1]:15s}: {cm[1]}")
