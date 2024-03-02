import torch
import pytorch_lightning as pl
import random
from transformers import DetrImageProcessor
import torchvision
import os


class SKUDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor, train=True):
        ann_file = os.path.join(img_folder, "custom_train_filtered.json" if train else "custom_val.json")
        super().__init__(img_folder, ann_file)
        self.processor = processor
    
    def prune_bounding_boxes(self, target, x_offset, y_offset, crop_size):
        x_bound = x_offset + crop_size
        y_bound = y_offset + crop_size
        
        pruned_targets = [{
            **obj, 
            'bbox': [obj['bbox'][0] - x_offset, obj['bbox'][1] - y_offset, obj["bbox"][2], obj["bbox"][3]]
        } for obj in target if x_offset <= obj['bbox'][0] < x_bound and y_offset <= obj['bbox'][1] < y_bound]
        return pruned_targets
    
    def __getitem__(self, idx):
        try:
            img, target = super().__getitem__(idx)
            img = torchvision.transforms.functional.to_tensor(img)
            crop_size = img.shape[-1] // 2
            random_index = random.choice(range(4))
            if random_index == 0:
                img_crop = img[:, :crop_size, :crop_size]
                x_offset, y_offset = 0, 0
            elif random_index == 1:
                img_crop = img[:, :crop_size, crop_size:]
                x_offset, y_offset = crop_size, 0
            elif random_index == 2:
                img_crop = img[:, crop_size:, :crop_size]
                x_offset, y_offset = 0, crop_size
            elif random_index == 3:
                img_crop = img[:, crop_size:, crop_size:]
                x_offset, y_offset = crop_size, crop_size
            # Prune bounding boxes
            target = self.prune_bounding_boxes(target, x_offset, y_offset, crop_size)
            image_id = self.ids[idx]

            selected_target = {'image_id': image_id, 'annotations': target}
            encoding = self.processor(
                images=torchvision.transforms.functional.to_pil_image(img_crop), 
                annotations=selected_target, 
                return_tensors="pt"
            )
            pixel_values = encoding["pixel_values"].squeeze()
            target_return = encoding["labels"][0] if len(encoding["labels"]) > 0 else torch.tensor([])
        except OSError:
            return None
        return pixel_values, target_return


class BatchCollator:
    def __init__(self):
        super().__init__()
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    def __call__(self, batch):
        pixel_values = [item[0] for item in batch if item is not None]
        encoding = self.processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch if item is not None]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch

def make_dataloader(dataset, phase, batch_size, num_workers):
    collator = BatchCollator()
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=phase != "train",
        shuffle=phase == "train", 
    )
    return data_loader


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, dataset_name):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    def setup(self, stage: str):
        if stage == "fit":
            self.data_train = SKUDetection("SKU110K_fixed/data", self.processor, True)
            self.data_val = SKUDetection("SKU110K_fixed/data", self.processor, False)

    def train_dataloader(self):
        return make_dataloader(self.data_train, "train", batch_size=self.batch_size, num_workers=6)
    
    def val_dataloader(self):
        return make_dataloader(self.data_val, "val", batch_size=self.batch_size, num_workers=2)
