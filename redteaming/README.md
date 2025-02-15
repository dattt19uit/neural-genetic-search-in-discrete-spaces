# RedTeaming Language Models
Code for NGS for red-teaming language models. The codebase is build upon [this work](https://openreview.net/forum?id=1mXufFuv95) [1].

## Installation
We recommend using Conda to manage the environment. To install the environment, run the following command:
```bash
conda env create -f environment.yml
conda activate ngs-redteam
```

## Supervised fine-tuning
First, fine-tune gpt2 model on the initial SFT dataset.

```bash
bash scripts/run_sft.sh
```

## GFlowNet fine-tuning with MLE-smoothing
Next, fine-tune gpt2 model with GFlowNet-MLE as proposed in the paper.

```bash
bash scripts/run_gfn_mle.sh {model}
```
where `{model}` is the model you want to train. The models used in the paper are "llama-3.2" (llama-3.2-3B-Instruct) and "llama-3.1" (llama-3.1-8B-Instruct).

## Evaluation
To evaluate a trained model with a specific decoding method, run the following command:
```bash
bash scripts/eval.sh {model} {seed}
```
The script includes evaluation of all decoding methods used in the paper.

## References
[1] Lee, Seanie, et al. "Learning diverse attacks on large language models for robust red-teaming and safety tuning." arXiv preprint arXiv:2405.18540 (2024).  
