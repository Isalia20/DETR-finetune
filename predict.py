import matplotlib.pyplot as plt
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageOps
import requests

def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for _, _, (xmin, ymin, xmax, ymax)  in zip(scores.tolist(), labels.tolist(), boxes.tolist()):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, linewidth=5, color='green'))
    plt.axis('off')
    plt.savefig("temp.jpg")

def main():
    url = "https://github.com/Isalia20/DETR-finetune/blob/main/IMG_3507.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    image = ImageOps.exif_transpose(image)

    # you can specify the revision tag if you don't want the timm dependency
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("isalia99/detr-resnet-50-sku110k")
    model = model.eval()
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.8)[0]

    plot_results(image, results["scores"], results["labels"], results["boxes"])
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
        )

if __name__ == "__main__":
    main()
  