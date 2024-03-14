import pandas as pd
from pathlib import Path
from typing import Dict, List
import json
import os
from PIL import Image

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

def convert_to_coco(annotations, image_folder_path):
    images = []
    coco_annotations = []
    categories = [{"id": 1, "name": "category_name", "supercategory": "none"}]

    annotation_id = 0
    for image_id, image_name in enumerate(annotations):
        # Get image dimensions
        image_path = os.path.join(image_folder_path, image_name)
        with Image.open(image_path) as img:
            width, height = img.size

        # Add image information
        images.append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": image_name
        })

        # Add annotation information
        for bbox in annotations[image_name]:
            x, y, x2, y2 = bbox
            width = x2 - x
            height = y2 - y
            area = width * height

            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x, y, width, height],
                "area": area,
                "iscrowd": 0
            })
            annotation_id += 1

    return {
        "images": images,
        "annotations": coco_annotations,
        "categories": categories
    }

def convert_pipeline(image_folder_path: str, split: str):
    annotations_path = f'annotations_{split}.json'
    with open(annotations_path) as f:
        annotations = json.load(f)
    coco_dataset = convert_to_coco(annotations, image_folder_path)
    with open(f'coco_annotations_{split}.json', 'w') as f:
        json.dump(coco_dataset, f)

def main(split: str):
    image_folder_path = 'SKU110K_fixed/images'
    df = read_anns(Path("SKU110K_fixed/annotations"), split)
    df.columns = ["image_name", "x_left", "y_top", "x_right", "y_bot", "class", "img_height", "img_width"]
    annotations = anns_to_dict(df)
    save_dict(annotations, f"annotations_{split}.json")
    convert_pipeline(image_folder_path, split)

if __name__ == "__main__":
    main("train")
    main("val")
    main("test")
    print("Conversion to COCO format completed!")
