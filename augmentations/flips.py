import torch
from torchvision.transforms.functional import hflip

def hflip_image_and_targets(image, target):
    target_boxes = target["boxes"]
    target_xs = target_boxes[:, [0]]
    target_xs = 1 - target_xs
    image = hflip(image)
    targets = torch.cat([target_xs, target_boxes[:, 1:]], dim=1)
    target["boxes"] = targets
    return image, target
