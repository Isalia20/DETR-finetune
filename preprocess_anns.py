import pandas as pd
from pathlib import Path
from typing import Dict, List
import json

def read_anns(annotation_path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(annotation_path / f"annotations_{split}.csv")
    return df

def anns_to_dict(df: pd.DataFrame):
    annotations = {}

    for i in df.itertuples():
        x_left = i.x_left
        y_top = i.y_top
        x_right = i.x_right
        y_bot = i.y_bot
        anns = [x_left, y_top, x_right, y_bot]
        img_name = i.image_name
        if img_name in annotations:
            annotations[img_name].append(anns)
        else:
            annotations[img_name] = [anns]
    return annotations

def save_dict(annotations: Dict[str, List[int]], json_name: str):
    with open(json_name, "w") as outfile:
        json.dump(annotations,outfile)

def main(split: str):
    df = read_anns(Path("SKU110K_fixed/annotations"), split)
    df.columns = ["image_name", "x_left", "y_top", "x_right", "y_bot", "class", "img_height", "img_width"]
    annotations = anns_to_dict(df)
    save_dict(annotations, f"annotations_{split}.json")

if __name__ == "__main__":
    main("train")
    main("val")
    main("test")
