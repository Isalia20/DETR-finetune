import torch
import pytorch_lightning as pl
from transformers import DetrForObjectDetection


class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, num_queries: int):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                            revision="no_timm",
                                                            num_labels=2,
                                                            num_queries=num_queries,
                                                            ignore_mismatched_sizes=True,
                                                            )
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict

    def training_step(self, batch, batch_idx):        
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss, on_epoch=True, on_step=False)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item(), on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True, on_step=False)
        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item(), on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):
        param_dicts = [
                {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": self.lr_backbone,
                },
        ]
        optimizer = torch.optim.AdamW(param_dicts, 
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20)
        return [optimizer], [lr_scheduler]
