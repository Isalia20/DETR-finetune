import torch
import pytorch_lightning as pl
from transformers import DetrForObjectDetection, DetrImageProcessor
from torch import nn
from transformers.image_transforms import center_to_corners_format
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from model_type_map import MODEL_TYPE_MAP


def replace_relu_with_silu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.SiLU())
        else:
            replace_relu_with_silu(child)

class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, num_queries: int, lr_decay_steps: int, **kwargs):
        super().__init__()
        self.lr_decay_steps = lr_decay_steps
        hf_model_id = MODEL_TYPE_MAP[kwargs["model_type"]]
        self.model = DetrForObjectDetection.from_pretrained(hf_model_id,
                                                            revision="no_timm" if kwargs["model_type"] == "resnet-50" else None,
                                                            num_labels=2,
                                                            num_queries=num_queries,
                                                            ignore_mismatched_sizes=True,
                                                            )
        self.processor = DetrImageProcessor.from_pretrained(hf_model_id)
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.load_pretrained_num_queries(kwargs["model_type"])
        if kwargs["train_backbone"]:
            for param in self.parameters():
                param.requires_grad = True
        if kwargs["replace_relu"]:
            replace_relu_with_silu(self.model)
        print(self.model)
        self.map_metric = MeanAveragePrecision(box_format="cxcywh", iou_type="bbox", class_metrics=False).to(torch.device("cuda"))
    
    def load_pretrained_num_queries(self, model_type):
        model_weights = {
            "resnet-50": "../detr-r50-e632da11.pth",
            "resnet101-dc5": "../detr-r101-dc5-a2e86def.pth"
        }
        if model_type in model_weights:
            weight_dict = torch.load(model_weights[model_type])
            self.query_weights = weight_dict["model"]["query_embed.weight"]
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        new_weights = torch.nn.Parameter(self.query_weights.clone())
        self.model.model.query_position_embeddings = nn.Embedding(num_embeddings=400, embedding_dim=256)
        noise = torch.randn_like(self.model.model.query_position_embeddings.weight) * 0.01
        for i in range(0, 400, 100):
            self.model.model.query_position_embeddings.weight.data[i:i+100] = new_weights
        self.model.model.query_position_embeddings.weight.data += noise

    def common_step(self, batch):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss, on_epoch=True, on_step=False)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item(), on_epoch=True, on_step=False)

        return loss
    
    def predict_image(self, batch):
        with torch.no_grad():
            outputs = self.model(pixel_values=batch["pixel_values"].cuda(), pixel_mask=batch["pixel_mask"].cuda())
        return outputs
    
    def update_map(self, batch):
        outputs = self.predict_image(batch)
    
        postprocessed_outputs = self.processor.post_process_object_detection(outputs,
                                                                             threshold=0.8)
        predictions = [
            {
                "boxes": postprocessed_output["boxes"].cuda(),
                "labels": torch.ones(postprocessed_output["boxes"].shape[0], device="cuda", dtype=torch.int),
                "scores": postprocessed_output["scores"].cuda(),
            } for postprocessed_output in postprocessed_outputs
        ]
        target = [center_to_corners_format(i["boxes"]).squeeze(0).cuda() for i in batch["labels"]]
        ground_truths = [
            {
                "boxes": box,
                "labels": torch.ones(box.shape[0], device="cuda", dtype=torch.int),
            } for box in target
        ]
        # Update the metric
        self.map_metric.update(predictions, ground_truths)

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch)
        self.update_map(batch)
        self.log("validation_loss", loss, on_epoch=True, on_step=False)
        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item(), on_epoch=True, on_step=False)
        return loss

    def on_validation_epoch_end(self):
        map_metric_dict = self.map_metric.compute()
        self.log_dict(map_metric_dict, on_epoch=True, on_step=False)
        self.map_metric.reset()
        
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
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.lr_decay_steps)
        return [optimizer], [lr_scheduler]
