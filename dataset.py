import torch
import pytorch_lightning as pl
import torchvision
import os
from augmentations.blur import BlurImage
from augmentations.flips import hflip_image_and_targets
from transformers import DetrImageProcessor
import random


class SKUDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor, train=True):
        ann_file = os.path.join(img_folder, "custom_train.json" if train else "custom_val.json")
        super().__init__(img_folder, ann_file)
        self.train = train
        self.processor = processor
        self.blur = BlurImage()

    def apply_blur(self, image):
        blur_bool = torch.randint(0, 2, (1,)).item()
        image = self.blur(image, blur_bool)
        return image
    
    def __getitem__(self, idx):
        while idx in [3577, 5350, 8089, 8136]:
            idx = random.randint(0, self.__len__() - 1)
        img, target = super().__getitem__(idx)
        # Prune bounding boxes
        image_id = self.ids[idx]
        selected_target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(
            images=img,
            annotations=selected_target,
            return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()
        target_return = encoding["labels"][0] if len(encoding["labels"]) > 0 else torch.tensor([])
        if self.train:
            img = self.apply_blur(img)
            pixel_values = encoding["pixel_values"].squeeze()
            target_return = encoding["labels"][0] if len(encoding["labels"]) > 0 else torch.tensor([])
            hflip_bool = torch.randint(0, 2, (1, )).item()
            if hflip_bool:
                pixel_values, target_return = hflip_image_and_targets(pixel_values, target_return)
        else:
            pixel_values = encoding["pixel_values"].squeeze()
            target_return = encoding["labels"][0] if len(encoding["labels"]) > 0 else torch.tensor([])
        return pixel_values, target_return



class BatchCollator:
    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def __call__(self, batch):
        pixel_values = [item[0] for item in batch if item is not None]
        encoding = self.processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch if item is not None]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch

def make_dataloader(dataset, phase, batch_size, num_workers, processor):
    collator = BatchCollator(processor=processor)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        shuffle=phase == "train", 
    )
    return data_loader


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, dataset_name, model_type):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.processor = DetrImageProcessor.from_pretrained(model_type)

    def setup(self, stage: str):
        if stage == "fit":
            self.data_train = SKUDetection("../SKU110K_fixed/data", self.processor, True)
            self.data_val = SKUDetection("../SKU110K_fixed/data", self.processor, False)

    def train_dataloader(self):
        return make_dataloader(self.data_train, "train", batch_size=self.batch_size, num_workers=6, processor=self.processor)
    
    def val_dataloader(self):
        return make_dataloader(self.data_val, "val", batch_size=self.batch_size, num_workers=2, processor=self.processor)
