$python = "c:/miniproject1.7/.venv/Scripts/python.exe"
& $python "c:/miniproject1.7/scripts/train_roast_classifier.py" --data-root "c:/miniproject1.7/artifacts/roast_cls_dataset" --epochs 20 --batch-size 16 --img-size 224 --device cpu
