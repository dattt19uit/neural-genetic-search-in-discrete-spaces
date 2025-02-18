# Genetic-guided GFlowNets

This repository provided implemented codes for the paper -- Genetic GFlowNets: Advancing in Practical Molecular Optimization Benchmark. 
> 

The codes are implemented our code based on the practical molecular optimization benchmark and Genetic-guided GFlowNets.

- [Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization (NeurIPS, 2022)](https://arxiv.org/abs/2206.12411)
(code: https://github.com/wenhao-gao/mol_opt)
- [Genetic-guided GFlowNets for Sample Efficient Molecular Optimization (NeurIPS, 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4b25c000967af9036fb9b207b198a626-Abstract-Conference.html)
(code: https://github.com/hyeonahkimm/genetic_gfn)



## Installation

```
conda create -n pmo python==3.7
conda activate pmo
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install PyTDC==0.4.0
pip install PyYAML
pip install rdkit
pip install selfies
pip install wandb
```


## Usage
```
CUDA_VISIBLE_DEVICES=0 python run.py ngs --task simple --oracle qed --seed 0
```


