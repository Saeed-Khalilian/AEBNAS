


# AEBNAS: Strengthening Exit Branches in Early-Exit Networks through Hardware-Aware Neural Architecture Search

## Abstract
Early-exit networks are effective solutions for reducing the overall energy consumption and latency of deep learning models by adjusting computation based on the complexity of input data. By incorporating intermediate exit branches into the architecture, they provide less computation for simpler samples, which is particularly beneficial for resource-constrained devices where energy consumption is crucial. However, designing early-exit networks is a challenging and time-consuming process due to the need to balance efficiency and performance. Recent works have utilized Neural Architecture Search (NAS) to design more efficient early-exit networks, aiming to reduce average latency while improving model accuracy by determining the best positions and number of exit branches in the architecture. Another important factor affecting the efficiency and accuracy of early-exit networks is the depth and types of layers in the exit branches. In this paper, we use hardware-aware NAS to strengthen exit branches, considering both accuracy and efficiency during optimization. Our performance evaluation on the CIFAR-10, CIFAR-100, and SVHN datasets demonstrates that our proposed framework, which considers varying depths and layers for exit branches along with adaptive threshold tuning, designs early-exit networks that achieve higher accuracy with the same or lower average number of MACs compared to the state-of-the-art approaches. 

## Installation

### Prerequisites
- Python 3.10
- PyTorch
- CUDA (optional, for GPU acceleration) 

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Download pre-trained supernets (optional):
```bash
# OFA supernet weights are provided in ofa_nets/
# Ensure pre-trained weights are available for faster search
```

## Early-Exit Model Design 

### CIFAR-10 Dataset

```bash
python3 AEBNAS.py \
    --sec_obj macs \
    --n_gpus 4 \
    --gpu 1 \
    --n_workers 4 \
    --n_epochs 5 \
    --dataset cifar10 \
    --n_classes 10 \
    --data ~/datasets \
    --supernet_path ./ofa_nets/ofa_eembv3 \
    --pretrained \
    --save ./results/cifar10 \
    --iterations 30 \
    --vld_size 5000 \
    --lr 32 \
    --ur 72 \
    --n_doe 100 \
    --target_macs 17
```

### SVHN Dataset

```bash
python3 AEBNAS.py \
    --sec_obj macs \
    --n_gpus 4 \
    --gpu 1 \
    --n_workers 4 \
    --n_epochs 5 \
    --dataset svhn \
    --n_classes 10 \
    --data ~/datasets \
    --supernet_path ./ofa_nets/ofa_eembv3 \
    --pretrained \
    --save ./results/svhn \
    --iterations 30 \
    --vld_size 7325 \
    --lr 32 \
    --ur 72 \
    --n_doe 100 \
    --target_macs 1
```

### CIFAR-100 Dataset

```bash
    python3 AEBNAS.py \
        --sec_obj macs \
        --n_gpus 1 \
        --gpu 1 \
        --n_workers 4 \
        --n_epochs 5 \
        --dataset cifar100 \
        --n_classes 100 \
        --data ~/datasets \
        --supernet_path ./ofa_nets/ofa_eembv3 \
        --pretrained \
        --save ./results/aebnas_cifar100 \
        --iterations 30 \
        --vld_size 5000 \
        --lr 32 \
        --ur 72 \
        --n_doe 100 \
        --target_macs 17 \
        --change_epochs_it 28 \
        --second_n_epochs 250
```

### Static Model Design (without early exit) 

Example: Standard NAS for static MobileNetV3 architecture on CIFAR-10 (without exit branches):

```bash
python3 AEBNAS.py \
    --sec_obj macs \
    --n_gpus 1 \
    --gpu 1 \
    --n_workers 4 \
    --n_epochs 5 \
    --dataset cifar10 \
    --n_classes 10 \
    --data ~/datasets \
    --supernet_path ./ofa_nets/ofa_mbv3_d234_e346_k357_w1.0 \
    --pretrained \
    --save ./results/static_cifar10 \
    --iterations 30 \
    --n_doe 100 \
    --vld_size 5000
``` 



## Citation
```bibtex
@inproceedings{robben2025aebnas,
  title={AEBNAS: Strengthening Exit Branches in Early-Exit Networks through Hardware-Aware Neural Architecture Search},
  author={Robben, Oscar and Khalilian, Saeed and Meratnia, Nirvana},
  booktitle={2025 3rd International Conference on Federated Learning Technologies and Applications (FLTA)},
  pages={580--587},
  year={2025},
  organization={IEEE}
}
```

## References
This work builds upon:
- [NSGANetV2](https://github.com/mikelzc1990/nsganetv2): Multi-objective NAS with surrogate optimization
- [Once-for-All](https://github.com/mit-han-lab/once-for-all): Efficient supernet-based NAS
- [EDANAS](https://github.com/AI-Tech-Research-Lab/EarlyExits.git): EDANAS: Adaptive Neural Architecture Search for Early Exit Neural Networks

