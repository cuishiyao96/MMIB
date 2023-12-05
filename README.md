# MMIB

Code for the the paper "Enhancing Multimodal Entity and Relation Extraction with Variational Information Bottleneck", which is accepted by TASLP2023


## Model Architecture

![]()


## Data

We followed [Chen et al.2023](https://github.com/zjunlp/HVPNeT) and [Zhang et al.2023](https://github.com/TransformersWsz/UMGF) for data processing.

## Commands to run the code:

CUDA_VISIBLE_DEVICES=1 python main.py --dataset_name twitter15 --num_epochs=30 --batch_size=8 --lr=3e-5 --warmup_ratio=0.03 --eval_begin_epoch=1 --seed=0 --ignore_idx=0 --max_seq=128  --log_name twitter15_model --do_train

CUDA_VISIBLE_DEVICES=1 python main.py --dataset_name twitter17 --num_epochs=30 --batch_size=8 --lr=3e-5 --warmup_ratio=0.03 --eval_begin_epoch=1 --seed=0 --ignore_idx=0 --max_seq=128  --log_name twitter17_model --do_train

CUDA_VISIBLE_DEVICES=1 python main.py --dataset_name MRE --num_epochs=30 --batch_size=8 --lr=3e-5 --warmup_ratio=0.03 --eval_begin_epoch=1 --seed=0 --ignore_idx=0 --max_seq=80  --log_name MRE_model --do_train