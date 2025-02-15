import logging
import os
from pathlib import Path
import time
from typing import Any

import lightning as L
from lightning.pytorch import loggers

from rl4co.models.zoo.gfacs import GFACS
from rl4co.models.zoo.ngs import NGS
from rl4co.envs import TSPEnv, CVRPEnv, PCTSPEnv, OPEnv
from rl4co.utils.trainer import RL4COTrainer


ENV_CLASS = {
    "tsp": TSPEnv,
    "cvrp": CVRPEnv,
    "pctsp": PCTSPEnv,
    "op": OPEnv,
}

MODEL_CLASS = {
    "aco": GFACS,
    "ngs": NGS,
    "sampling": GFACS,  # GFACS without pheromone update
}

BETAS = {
    "tsp": (200, 1000, 5),
    "cvrp": (500, 2000, 5),
    "pctsp": (50, 200, 5),
    "op": (5, 20, 5),
}


def get_env(env_name: str, scale: int) -> TSPEnv | CVRPEnv | PCTSPEnv | OPEnv:
    generator_params: dict[str, Any] = {"num_loc": scale}
    match env_name:
        case "tsp":
            val_file = f"tsp/tsp{scale}_deepaco_val_seed42.npz"
            test_file = f"tsp/deepaco_tsp{scale}.npz"
        case "cvrp":
            generator_params.update({"capacity": 50.0})
            val_file = f"vrp/vrp{scale}_deepaco_val_seed42.npz"
            test_file = f"vrp/deepaco_vrp{scale}.npz"
        case "pctsp":
            val_file = f"pctsp/pctsp{scale}_val_seed4321.npz"
            test_file = f"pctsp/pctsp{scale}_test_seed1234.npz"
        case "op":
            val_file = f"op/op_dist{scale}_val_seed4321.npz"
            test_file = f"op/op_dist{scale}_test_seed1234.npz"
        case _:
            raise ValueError(f"Unknown environment: {env_name}")

    env_params = {
        "check_solution": False,
        "generator_params": generator_params,
        "val_file": val_file,
        "test_file": test_file,
    }
    return ENV_CLASS[env_name](**env_params)


def get_model(
    env_name: str, method: str, scale: int, is_train: bool, test_n_iter: int
) -> GFACS | NGS:
    env = get_env(env_name, scale)

    rank_coefficient = 0.001
    n_off = 100
    tsp_max_iterations = scale // 4 if is_train else scale
    tsp_n_perturbations = 2 if is_train else 5

    # common configs
    model_config: dict[str, Any] = {
        "env": env,
        "batch_size": 20 if scale <= 500 else 10,
        "val_batch_size": 20 if scale <= 500 else 10,
        "test_batch_size": 32 if scale <= 500 else 16,
        "train_data_size": 400,
        "metrics": {"test": [f"reward_{i:03d}" for i in range(test_n_iter)]},
        "optimizer": "AdamW",
        "optimizer_kwargs": {"lr": 5e-4, "weight_decay": 0},
        "lr_scheduler": "CosineAnnealingLR",
        "lr_scheduler_kwargs": {"T_max": 50, "eta_min": 1e-5},
    }

    policy_kwargs: dict[str, Any] = {
        "n_iterations": {"train": 1, "val": 1, "test": test_n_iter},
    }

    is_aco = method in ["aco", "sampling"]
    alg_kwargs_key = "aco_kwargs" if is_aco else "ga_kwargs"
    alg_kwargs = {}

    # env specific configs
    match env_name:
        case "tsp":
            model_config.update({"train_with_local_search": True})
            policy_kwargs.update({"k_sparse": scale // 10})
            alg_kwargs.update(
                {
                    "use_local_search": True,
                    "use_nls": True,
                    "local_search_params": {"max_iterations": tsp_max_iterations, "num_threads": 32},
                    "n_perturbations": tsp_n_perturbations,
                    "perturbation_params": {"max_iterations": 20, "num_threads": 32},
                }
            )
        case "cvrp":
            model_config.update({"train_with_local_search": True})
            policy_kwargs.update({"k_sparse": scale // 5})
            alg_kwargs.update(
                {
                    "use_local_search": True,
                    "use_nls": True,
                    "n_perturbations": 1,
                    "local_search_params": {"max_iterations": scale * 5},
                    "perturbation_params": {"max_iterations": 5},
                }
            )
        case "pctsp":  # do not use local search since the performance gain is marginal
            model_config.update({"train_with_local_search": False})
            policy_kwargs.update({"k_sparse": scale // 5})
            alg_kwargs.update({"use_local_search": False})
        case "op":  # do not use local search since the performance gain is marginal
            model_config.update({"train_with_local_search": False})
            policy_kwargs.update({"k_sparse": scale // 5})
            alg_kwargs.update({"use_local_search": False})
        case _:
            raise ValueError(f"Unknown environment: {env_name}")

    # model specific configs
    match method:
        case "aco":
            policy_kwargs.update({"n_ants": {"train": 30, "val": 30, "test": n_off}})
            beta_min, beta_max, beta_flat_epochs = BETAS[env_name]
            model_config.update(
                {
                    "beta_min": beta_min, "beta_max": beta_max, "beta_flat_epochs": beta_flat_epochs
                }
            )
        case "ngs":
            policy_kwargs.update({"n_population": {"test": 100}, "n_offspring": {"test": n_off}})
            alg_kwargs.update({"rank_coefficient": rank_coefficient})
        case "sampling":
            policy_kwargs.update({"n_ants": {"test": n_off}})
            alg_kwargs.update({"update_pheromone": False})
        case _:
            raise ValueError(f"Unknown model: {method}")

    policy_kwargs.update({alg_kwargs_key: alg_kwargs})
    model_config.update({"policy_kwargs": policy_kwargs})

    return MODEL_CLASS[method](**model_config)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, choices=["tsp", "cvrp", "pctsp", "op"],)
    parser.add_argument("scale", type=int, choices=[200, 500, 1000])
    parser.add_argument("mode", type=str, choices=["train", "test"])
    parser.add_argument("--method", type=str, default=None, choices=[None, "aco", "ngs", "sampling"])
    parser.add_argument("--seed", type=int, default=0, choices=range(5))
    parser.add_argument("--exp_name", type=str, default="default", help="Experience name")
    parser.add_argument("--device", type=int, default=None, help="Device")
    args = parser.parse_args()

    is_train = args.mode == "train"
    test_n_iter = 100 if not is_train else 10
    if is_train:
        assert args.method is None
        args.method = "aco"  # Use GFACS for training
    else:
        assert args.method is not None

    log_dir = f"{Path(__file__).parent}/results/{args.env_name}/{args.scale}"
    version = f"seed{args.seed}/{'train' if is_train else 'test_' + args.method + '_iter' + str(test_n_iter)}"

    exp_dir = f"{log_dir}/{args.exp_name}/{version}"
    os.makedirs(exp_dir, exist_ok=True)
    filelog = f"{exp_dir}/log.txt"

    L.seed_everything(args.seed, workers=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    fileHandler = logging.FileHandler(filelog)
    stdoutHandler = logging.StreamHandler()
    fileHandler.setFormatter(logFormatter)
    stdoutHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    logger.addHandler(stdoutHandler)

    model = get_model(args.env_name, args.method, args.scale, is_train, test_n_iter)
    precision = "bf16-mixed"
    metric_logger = loggers.CSVLogger(log_dir, name=args.exp_name, version=version, flush_logs_every_n_steps=10)
    trainer = RL4COTrainer(
        max_epochs=50,
        gradient_clip_val=3,
        devices=[args.device] if args.device is not None else "auto",
        precision=precision,
        logger=metric_logger,
    )

    if is_train:
        logger.info("... Training ...")
        trainer.fit(model)
        # the checkpoint is automatically saved to the f"{log_dir}/{exp_name}/seed{args.seed}/train/checkpoints"

    ckpt_dir = f"{log_dir}/{args.exp_name}/seed{args.seed}/train/checkpoints"
    if len(os.listdir(ckpt_dir)) == 0:
        raise Exception("No checkpoint found!")
    elif len(os.listdir(ckpt_dir)) > 1:
        raise Exception("Multiple checkpoints found!")

    logger.info(f"... Testing {args.method} with n_iter={test_n_iter} ...")
    start = time.time()
    trainer.test(model, ckpt_path=os.path.join(ckpt_dir, os.listdir(ckpt_dir)[0]))
    end = time.time()

    logger.info("Done!")
    logging_points = [1, 3, 10, 30, 100]
    for i in logging_points:
        if i > test_n_iter:
            break
        logger.info(
            f"Iteration {i:3d}:\t{trainer.logged_metrics[f'test/reward_{i-1:03d}'].item():.3f}"
        )
    logger.info(f"Elapsed time :\t{end - start:.2f}s")
