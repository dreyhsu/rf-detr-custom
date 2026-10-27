import random, shutil

from pathlib import Path
 
SRC = Path(r"D:\Dre\PDE_yolo_infra\train_dataset\hph_packing_merged_aug")        # has images/ labels/ classes.txt

DST = Path(r"D:\Dre\rf-detr\dateset")

VAL_RATIO = 0.2

random.seed(42)
 
images = [p for p in (SRC / "images").iterdir()

          if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

random.shuffle(images)

n_val = int(len(images) * VAL_RATIO)

splits = {"valid": images[:n_val], "train": images[n_val:]}
 
for split, files in splits.items():

    (DST / split / "images").mkdir(parents=True, exist_ok=True)

    (DST / split / "labels").mkdir(parents=True, exist_ok=True)

    for img in files:

        shutil.copy(img, DST / split / "images" / img.name)

        lbl = SRC / "labels" / (img.stem + ".txt")

        if lbl.exists():   # images with no objects can have no label file

            shutil.copy(lbl, DST / split / "labels" / lbl.name)
 
names = [l.strip() for l in (SRC / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]

yaml = "train: train/images\nval: valid/images\n\n"

yaml += f"nc: {len(names)}\nnames: {names}\n"

(DST / "data.yaml").write_text(yaml, encoding="utf-8")

print({k: len(v) for k, v in splits.items()}, names)
 