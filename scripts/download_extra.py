import os, kaggle

jobs = [
    ("noveliopi/coffee-bean-roasting-level", "data/raw/roast_extra1"),
    ("ardiyanto24/coffee-bean-classification-dataset", "data/raw/roast_extra2"),
    ("sujitraarw/coffee-green-bean-with-17-defects-original", "data/raw/defect"),
]
for ref, path in jobs:
    os.makedirs(path, exist_ok=True)
    print(f"=> {ref} -> {path}", flush=True)
    kaggle.api.dataset_download_files(ref, path=path, unzip=True, quiet=False)
    print(f"   done. files in root:", len(os.listdir(path)), flush=True)
