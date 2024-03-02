# DETR-finetune
Repository for finetuning DETR on SKU110K dataset. Checkpoints will be released once training is done. 

Finetuning model with num_queries!=100 is quite hard since it requires retraining on the whole COCO dataset from scratch. I have tried finetuning with num_queries=400 but the results achieved were abysmal. Plus even training on just a single image with num_queries=400 seemed to be impossible for some odd reason. Therefore this repository takes following 
strategy for finetuning and then for inference:
1. Take an image and divide it into 4 parts(4 equal crops)
2. Train this way
3. During inference crop image into 4 equal parts do inference with num_queries=100 for each part separately
4. Stitch the predictions together

While this kind of setup is suboptimal, I'm still researching why training with num_queries=400 doesn't work.