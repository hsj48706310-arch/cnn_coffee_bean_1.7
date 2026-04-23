"""
data/raw/<task>/.../<class>/*.jpg 를 스캔해서
data/annotations/<task>.csv 와 data/splits/<task>_{train,val,test}.csv 생성.

Stratified split (클래스 비율 유지). Kaggle 데이터는 'session_id' 개념이
없으므로 GroupSplit은 사용하지 않음.

사용:
    python scripts/make_splits.py --task roast
    python scripts/make_splits.py --task defect --classes normal defective
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_class_dir(raw_root: Path, cls: str) -> list[Path]:
    """raw_root 하위에서 폴더명이 cls(대소문자 무시)인 디렉토리를 모두 찾는다."""
    matches = []
    for p in raw_root.rglob("*"):
        if p.is_dir() and p.name.lower() == cls.lower():
            matches.append(p)
    return matches


def collect_images(raw_root: Path, classes: list[str], task: str) -> pd.DataFrame:
    rows = []
    for cls in classes:
        dirs = find_class_dir(raw_root, cls)
        if not dirs:
            print(f"[!] class '{cls}' 폴더를 {raw_root} 아래에서 찾지 못함")
            continue
        for d in dirs:
            for img in d.rglob("*"):
                if img.suffix.lower() in IMG_EXT and img.is_file():
                    rows.append({
                        "path": img.resolve().as_posix(),
                        f"{task}_label": cls.lower(),
                        "source": d.relative_to(raw_root).parts[0]
                                  if d != raw_root else "root",
                    })
    df = pd.DataFrame(rows)
    return df


def stratified_split(df: pd.DataFrame, label_col: str, seed: int = 42):
    train, temp = train_test_split(df, test_size=0.30, stratify=df[label_col],
                                   random_state=seed)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp[label_col],
                                 random_state=seed)
    return train, val, test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["roast", "defect"])
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out_ann", default="data/annotations")
    p.add_argument("--out_split", default="data/splits")
    p.add_argument("--classes", nargs="+", default=None,
                   help="클래스 목록. 미지정 시 task별 기본값 사용")
    p.add_argument("--max_samples", type=int, default=None,
                   help="총 샘플 수 상한. stratified 다운샘플링.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    default_classes = {
        "roast":  ["green", "light", "medium", "dark"],
        "defect": ["normal", "defective"],
    }
    classes = args.classes or default_classes[args.task]
    raw_root = Path(args.raw) / args.task

    print(f"[+] scan: {raw_root}  classes={classes}")
    df = collect_images(raw_root, classes, args.task)
    if df.empty:
        raise SystemExit("[X] 수집된 이미지가 없습니다. 데이터 경로/클래스명을 확인하세요.")

    print("[+] class distribution:")
    print(df[f"{args.task}_label"].value_counts())

    if args.max_samples and len(df) > args.max_samples:
        n = args.max_samples
        frac = n / len(df)
        parts = []
        for _, g in df.groupby(f"{args.task}_label"):
            k = max(1, int(round(len(g) * frac)))
            parts.append(g.sample(k, random_state=args.seed))
        df = pd.concat(parts, ignore_index=True)
        print(f"[+] downsampled to {len(df)} rows (target {n})")
        print(df[f"{args.task}_label"].value_counts())

    Path(args.out_ann).mkdir(parents=True, exist_ok=True)
    Path(args.out_split).mkdir(parents=True, exist_ok=True)

    ann_path = Path(args.out_ann) / f"{args.task}.csv"
    df.to_csv(ann_path, index=False)
    print(f"[OK] annotation -> {ann_path} ({len(df)} rows)")

    train, val, test = stratified_split(df, f"{args.task}_label", args.seed)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        out = Path(args.out_split) / f"{args.task}_{name}.csv"
        part.to_csv(out, index=False)
        print(f"[OK] {name:5s} -> {out} ({len(part)} rows)")


if __name__ == "__main__":
    main()
