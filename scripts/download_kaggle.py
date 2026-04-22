"""
Kaggle 데이터셋 자동 다운로드.

선행 작업:
    1) https://www.kaggle.com/settings  ->  Create New API Token
    2) kaggle.json 을  %USERPROFILE%\.kaggle\kaggle.json  에 저장
    3) pip install kaggle
사용:
    python scripts/download_kaggle.py
    python scripts/download_kaggle.py --task defect
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

# 후보 데이터셋 (실제 ID는 변경 가능)
DATASETS = {
    "roast": [
        # (kaggle_dataset_id, 저장될 하위 폴더)
        ("gpiosenka/coffee-bean-dataset-resized-224-x-224", "gpiosenka"),
    ],
    "defect": [
        # USK-Coffee 류는 Kaggle에 직접 없을 수 있어 자리만 잡아둠
        # 필요 시 Roboflow/직접 업로드 데이터셋으로 교체
    ],
}


def download(task: str, out_root: Path) -> None:
    import kaggle  # 지연 임포트 (kaggle.json 없어도 import 시점 에러 회피용)

    target_root = out_root / task
    target_root.mkdir(parents=True, exist_ok=True)

    items = DATASETS.get(task, [])
    if not items:
        print(f"[!] '{task}' 에 등록된 Kaggle 데이터셋이 없습니다. "
              f"DATASETS 딕셔너리에 추가하세요.")
        return

    for ds_id, sub in items:
        out_dir = target_root / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[+] downloading {ds_id} -> {out_dir}")
        kaggle.api.dataset_download_files(ds_id, path=out_dir.as_posix(),
                                          unzip=True, quiet=False)

    print(f"[OK] '{task}' 다운로드 완료. 저장 위치: {target_root}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["roast", "defect"], default="roast")
    p.add_argument("--out", default="data/raw")
    args = p.parse_args()
    download(args.task, Path(args.out))


if __name__ == "__main__":
    main()
