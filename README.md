# MMIB

Code for the the paper "Enhancing Multimodal Entity and Relation Extraction with Variational Information Bottleneck", which is accepted by TASLP2023


## Model Architecture

![MMIB Model](https://raw.githubusercontent.com/cuishiyao96/MMIB/main/Figures/model.png)


## Data

We followed [Chen et al.2023](https://github.com/zjunlp/HVPNeT) and [Zhang et al.2023](https://github.com/TransformersWsz/UMGF) for data processing.

## Commands to run the code:

CUDA_VISIBLE_DEVICES=1 python main.py --dataset_name twitter15 --num_epochs=30 --batch_size=8 --lr=3e-5 --warmup_ratio=0.03 --eval_begin_epoch=1 --seed=0 --ignore_idx=0 --max_seq=128  --log_name twitter15_model --do_train

CUDA_VISIBLE_DEVICES=1 python main.py --dataset_name twitter17 --num_epochs=30 --batch_size=8 --lr=3e-5 --warmup_ratio=0.03 --eval_begin_epoch=1 --seed=0 --ignore_idx=0 --max_seq=128  --log_name twitter17_model --do_train

CUDA_VISIBLE_DEVICES=1 python main.py --dataset_name MRE --num_epochs=30 --batch_size=8 --lr=3e-5 --warmup_ratio=0.03 --eval_begin_epoch=1 --seed=0 --ignore_idx=0 --max_seq=80  --log_name MRE_model --do_train


## Citation

If our paper is helpful for you, please cite as (The TASLP citation is coming soon):

```
@article{DBLP:journals/corr/abs-2304-02328,
  author       = {Shiyao Cui and
                  Jiangxia Cao and
                  Xin Cong and
                  Jiawei Sheng and
                  Quangang Li and
                  Tingwen Liu and
                  Jinqiao Shi},
  title        = {Enhancing Multimodal Entity and Relation Extraction with Variational
                  Information Bottleneck},
  journal      = {CoRR},
  volume       = {abs/2304.02328},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2304.02328},
  doi          = {10.48550/ARXIV.2304.02328},
  eprinttype    = {arXiv},
  eprint       = {2304.02328},
  timestamp    = {Mon, 17 Apr 2023 15:20:10 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2304-02328.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```