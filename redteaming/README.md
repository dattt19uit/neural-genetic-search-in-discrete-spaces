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
bash scripts/run_gfn_mle.sh {source_victim}
```
where `{source_victim}` is the model you want to train. The models used in the paper are "llama-3.2" (Llama-3.2-3B-Instruct) and "llama-3.1" (Llama-3.1-8B-Instruct).

## Evaluation
To evaluate a trained model with a specific decoding method, run the following command:
```bash
bash scripts/eval.sh {source_victim} {seed}
```
The script runs all decoding methods used in the paper sequentially.

### Transfer
To evaluate the transfer performance (i.e., the target victim model is different from the source victim model), simply include the target model name as the third argument:
```bash
bash scripts/eval.sh {source_victim} {seed} {target_victim}
```
The target victim models considered in the paper are:
- "llama-3.2" (meta-llama/Llama-3.2-3B-Instruct)
- "llama-3.1" (meta-llama/Meta-Llama-3.1-8B-Instruct)
- "llama-3.3" (meta-llama/Llama-3.3-70B-Instruct)  # This model requires a large amount of memory (e.g., at least 2 x A100l-80GB)
- "gemma-2-9b" (google/gemma-2-9b-it)
- "qwen" (Qwen/Qwen2.5-7B-Instruct)
- "phi-4" (microsoft/phi-4)

## References
[1] Lee, Seanie, et al. "Learning diverse attacks on large language models for robust red-teaming and safety tuning." arXiv preprint arXiv:2405.18540 (2024).  
