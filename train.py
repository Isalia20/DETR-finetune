import pytorch_lightning as pl
import torch
from detr_model import Detr
from dataset import DataModule
from pytorch_lightning.callbacks import ModelCheckpoint


def main(batch_size, experiment_name):
    sku_data_module = DataModule(batch_size=batch_size, dataset_name="SKU110K")
    output_folder = f"checkpoints/{experiment_name}"
    callbacks = [
            ModelCheckpoint(monitor='validation_loss',
                            dirpath=output_folder + "/",
                            filename="{epoch}-{step}-{validation_loss:.3f}",
                            mode="min",
                            every_n_epochs=1,
                            save_top_k=3,
			                save_last=True,
                            ),
        ]
    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, num_queries=100)
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=300,
        precision=16,
        benchmark=True,
        callbacks=callbacks,
        # limit_val_batches=0.0,
    )
    trainer.fit(model,
                sku_data_module,
                )

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    main(8, "detr_train_100_queries_pretrained_sku_4_crops")
