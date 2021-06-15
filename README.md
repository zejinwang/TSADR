# TSADR
Official PyTorch code for our paper "Temporal Spatial-Adaptive Interpolation with Deformable Refinement for Electron Microscopic Images"[[arXiv]](https://arxiv.org/abs/2101.06771)

The cremi_triplet dataset which include three sub-datasets are available [[here]](https://drive.google.com/file/d/1bmwArABD4iifogokdyN8srIMY2_Qnf4S/view?usp=sharing). 
```
python train_sem.py --train_dir /your/train/path --val_dir /your/valid/path --batch_size 2 --num_epoch 100 --gpu_id 1 --out_dir /your/output/path
