import json
import os
from PIL import Image

# Define the paths to your datasets
train_annotations_path = 'annotations_train.json'
val_annotations_path = 'annotations_val.json'

# Load your datasets
with open(train_annotations_path) as f:
    train_annotations = json.load(f)
with open(val_annotations_path) as f:
    val_annotations = json.load(f)

# Define the path to the image folder
image_folder_path = 'SKU110K_fixed/images'

# This function will convert your dataset into COCO format
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
                "category_id": 1,  # Assuming a single category with ID 1
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

# Convert datasets
coco_train = convert_to_coco(train_annotations, image_folder_path)
coco_val = convert_to_coco(val_annotations, image_folder_path)

# Save the COCO formatted datasets
with open('coco_annotations_train.json', 'w') as f:
    json.dump(coco_train, f)

with open('coco_annotations_val.json', 'w') as f:
    json.dump(coco_val, f)

print("Conversion to COCO format completed!")