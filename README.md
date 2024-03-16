# DETR-finetune

Welcome to the repository dedicated to finetuning DETR on the SKU110K dataset! 🚀

We are excited to share that our trained checkpoint, DETR-Resnet-50 configured for SKU110K with 400 queries, is now available on HuggingFace. 🎉

## Why is this important?
Finetuning DETR with a `num_queries` parameter different from the default (100) is challenging. Our experiments show that without a proper initialization strategy, training tends to fail — resulting in low mean Average Precision (mAP) even after extensive training.

## Our Solution ✨
We discovered that using pretrained weights to initialize `num_queries` significantly improves performance. It leverages the pretrained model's ability to detect objects across various image areas, making a small adjustment to specialize in the new dataset much easier than starting from scratch.

🔍 Explore the `load_pretrained_num_queries` function in `detr_model.py` to see how we implement this strategy.

## Overcoming Initialization Challenges
When `num_queries` exceeds 400, we've found that duplicating the pretrained `num_queries=100` weights four times and introducing minor noise sets the stage for success.

## Results 📈
After extensive experimentation with various configurations, this approach stood out, achieving a 59.0 mAP on the SKU110K validation set.
