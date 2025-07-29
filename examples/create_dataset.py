from pathlib import Path
from dataclasses import dataclass
import tyro
import json
import random

@dataclass
class Args:
    # dataset: str
    data_dir: str
    train_split: float = 0.9

if __name__ == "__main__":
    args = tyro.cli(Args)

    data_dir = Path(args.data_dir)
    name = data_dir.stem

    output_dir = Path("./datasets") / name
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = data_dir / "train/ours_30000/gt"
    render_dir = data_dir / "train/ours_30000/renders"

    gt_images = list(gt_dir.glob("*.png"))
    random.shuffle(gt_images)
    total_images = len(gt_images)

    train_images = gt_images[:int(total_images * args.train_split)]
    test_images = gt_images[int(total_images * args.train_split):]

    # make dataset json
    dataset = {
        "train": {},
        "test": {}
    }

    for image in train_images:
        dataset["train"][image.stem] = {
            "image": str((render_dir / image.name).resolve()),
            "target_image": str((gt_dir / image.name).resolve()),
            "prompt": "remove degradation"
        }

    for image in test_images:
        dataset["test"][image.stem] = {
            "image": str((render_dir / image.name).resolve()),
            "target_image": str((gt_dir / image.name).resolve()),
            "prompt": "remove degradation"
        }

    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset, f, indent=4)