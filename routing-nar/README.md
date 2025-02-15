# Routing Problems: Non-AutoRegressive Models
Code for NGS for routing problems with non-autoregressive models (GFACS [1]). The codebase is build upon [rl4co](https://github.com/ai4co/rl4co) [2].

## Installation
Before installing the requirements, make sure you have created and activated a virtual environment.

```bash
pip install torch  # Install PyTorch first with proper CUDA version
pip install torch_geometric  # Install PyTorch Geometric with proper CUDA version
pip install -r requirements.txt

# Install the HGS-CVRP to use SWAP* local search
cd rl4co/envs/routing/cvrp
git clone git@github.com:ai4co/HGS-CVRP.git  # or git clone https://github.com/ai4co/HGS-CVRP.git
cd HGS-CVRP
bash build.sh
cd ../../../../../
```


## Training
To train a model, run the following command:

```bash
python run.py {problem} {scale} train --seed {seed} --device {device}
```

where,
- `{problem}` is the routing problem to solve. Choices: {"tsp", "cvrp", "pctsp", "op"}
- `{scale}` is the number of nodes in the problem instance. Choices: {200, 500, 1000}
- `{seed}` is the random seed. Choices: {0, 1, 2, 3, 4}
- `{device}` is the gpu index. If not provided, the code will automatically use the gpu if available.

## Evaluation
To evaluate a trained model with a specific search method, run the following command:

```bash
python run.py {problem} {scale} test --method {method} --seed {seed} --device {device}
```

where, `{problem}`, `{scale}`, `{seed}`, and `{device}` are the same as in the training command, and `{method}` is the method to evaluate. Choices: {"sampling", "aco", "ngs"}.


### TSPLib & CVRPLib
To evaluate the model on TSPLib and CVRPLib instances, run the following command:

```bash
python test_lib.py {problem} {scale} --method {method} --seed {seed} --device {device}
```

where, `{problem}` can be "tsp" or "cvrp", and `{scale}`, `{seed}`, `{device}`, and `{method}` are the same as in the evaluation command.


## References
[1] Kim, Minsu, et al. "Ant colony sampling with gflownets for combinatorial optimization." arXiv preprint arXiv:2403.07041 (2024).  
[2] Berto, Federico, et al. "Rl4co: an extensive reinforcement learning for combinatorial optimization benchmark." arXiv preprint arXiv:2306.17100 (2023).  