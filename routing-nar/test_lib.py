import logging
import os
from pathlib import Path
import pickle

import lightning as L
from tensordict import TensorDict
import torch

from rl4co.models.zoo.gfacs import GFACS
from rl4co.envs import TSPEnv, CVRPEnv

from run import get_model


def load_tsplib_dataset(env: TSPEnv, n_nodes, device):
    scale_map = {200: ("100", "299"), 500: ("300", "699"), 1000: ("700", "1499")}

    tsplib_dir = Path(__file__).parent / f"data/tsp/tsplib"
    problem_file = tsplib_dir / f"tsplib_{'_'.join(scale_map[n_nodes])}.pkl"
    optimum_file = tsplib_dir / f"optimum_{'_'.join(scale_map[n_nodes])}.csv"

    if not os.path.isfile(problem_file) or not os.path.isfile(optimum_file):
        raise FileNotFoundError(
            f"File(s) not found, please download the test dataset from the original repository."
        )

    with open(problem_file, "rb") as f:
        tsplib_list = pickle.load(f)

    with open(optimum_file, "r") as f:
        optimums = f.readlines()[1:]

    td_list = []
    scale_list = []
    name_list = []
    optimum_list = []
    for (locs, scale, name), line in zip(tsplib_list, optimums):
        name2, opt = line.strip().split(",")
        assert name.item() == name2
        td = env.reset(
            TensorDict({"locs": locs.unsqueeze(0)}, batch_size=[1])
        ).to(device)
        # td = lib_to_td(locs, device)
        td_list.append(td)
        scale_list.append(scale)
        name_list.append(name.item())
        optimum_list.append(int(opt))
    return td_list, scale_list, name_list, optimum_list


def load_vrplib_dataset(env: CVRPEnv, n_nodes, device, dataset_name="X"):
    assert dataset_name == "X"
    scale_map = {200: "100_299", 500: "300_699", 1000: "700_1001"}

    vrplib_dir = Path(__file__).parent / f"data/vrp/vrplib"
    problem_file = vrplib_dir / f"vrplib_{dataset_name}_{scale_map[n_nodes]}.pkl"
    optimum_file = vrplib_dir / f"vrplib_{dataset_name}_{scale_map[n_nodes]}_optimal.csv"

    if not os.path.isfile(problem_file) or not os.path.isfile(optimum_file):
        raise FileNotFoundError(
            f"File(s) not found, please download the test dataset from the original repository."
        )

    with open(problem_file, "rb") as f:
        vrplib_list = pickle.load(f)

    with open(optimum_file, "r") as f:
        optimums = f.readlines()[1:]

    td_list = []
    int_dist_list = []
    name_list = []
    optimum_list = []
    for (normed_demand, position, distance, name), line in zip(vrplib_list, optimums):
        # demand is already normalized by capacity
        # normalize the position and distance into [0.01, 0.99] range
        name2, opt = line.strip().split(",")
        assert name == name2

        scale = (position.max(0) - position.min(0)).max() / 0.98
        position = position - position.min(0)
        position = position / scale + 0.01

        # convert all to torch
        normed_demand = torch.tensor(normed_demand, device=device, dtype=torch.float32)
        position = torch.tensor(position, device=device, dtype=torch.float32)

        td = env.reset(
            TensorDict(
                {
                    "locs": position[1:].unsqueeze(0),
                    "depot": position[0].unsqueeze(0),
                    "demand": normed_demand[1:].unsqueeze(0),
                    "capacity": torch.tensor([[1.0]]),
                },
                batch_size=[1],
            ).to(device)
        )
        td_list.append(td)
        int_dist_list.append(distance)
        name_list.append(name)
        optimum_list.append(int(opt))

    return td_list, int_dist_list, name_list, optimum_list


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, choices=["tsp", "cvrp"],)
    parser.add_argument("scale", type=int, choices=[200, 500, 1000])
    parser.add_argument("--method", type=str, choices=["aco", "ngs", "sampling"])
    parser.add_argument("--seed", type=int, default=0, choices=range(5))
    parser.add_argument("--exp_name", type=str, default="default", help="Experience name")
    parser.add_argument("--device", type=int, default=None, help="Device")
    args = parser.parse_args()

    test_n_iter = 10
    log_dir = f"{Path(__file__).parent}/results/{args.env_name}/{args.scale}"
    version = f"seed{args.seed}/test_lib_{args.method}_iter{test_n_iter}"

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

    ckpt_dir = f"{log_dir}/{args.exp_name}/seed{args.seed}/train/checkpoints"
    if len(os.listdir(ckpt_dir)) == 0:
        raise Exception("No checkpoint found!")
    elif len(os.listdir(ckpt_dir)) > 1:
        raise Exception("Multiple checkpoints found!")

    # ckpt_model = GFACS.load_from_checkpoint(os.path.join(ckpt_dir, os.listdir(ckpt_dir)[0]))
    ckpt_model = GFACS.load_from_checkpoint(
        os.path.join(ckpt_dir, os.listdir(ckpt_dir)[0]),
        map_location=lambda storage, loc: storage.cuda(0)
    )

    ckpt = ckpt_model.policy.state_dict()

    device = "cpu" if args.device is None else f"cuda:{args.device}"
    model = get_model(args.env_name, args.method, args.scale, False, test_n_iter)
    model.policy.load_state_dict(ckpt)
    del ckpt_model, ckpt

    ### Load Lib dataset
    logger.info(f"... Testing {args.method} with n_iter={test_n_iter} ...")

    with torch.no_grad():
        model = model.to(device).eval()

        if args.env_name == "tsp":
            td_list, scale_or_dist_list, name_list, optimum_list = load_tsplib_dataset(
                model.env, args.scale, device  # type: ignore
            )
        elif args.env_name == "cvrp":
            td_list, scale_or_dist_list, name_list, optimum_list = load_vrplib_dataset(
                model.env, args.scale, device  # type: ignore
            )
        else:
            raise ValueError("Invalid env_name")

        # Re-calculate the cost using integer distances
        real_costs = dict()
        gaps = []
        for td, scale_or_dist, name, optimum in zip(td_list, scale_or_dist_list, name_list, optimum_list):
            output = model.policy(td, model.env, phase="test")

            if args.env_name == "tsp":
                locs = td["locs"].squeeze(0).cpu() * scale_or_dist
                distances = torch.cdist(locs, locs, p=2).ceil().int()
                u = output["actions"].cpu().squeeze(0)
                v = torch.roll(u, shifts=-1, dims=-1)
                cost = distances[u, v].sum().item()
            else:  # args.env_name == "cvrp"
                # add zero to the beginning of the tour
                u = torch.cat((torch.zeros(1, dtype=torch.long), output["actions"].cpu().squeeze(0)))
                v = torch.roll(u, shifts=-1, dims=-1)
                cost = scale_or_dist[u, v].sum().item()

            real_costs[name] = cost
            gap = (cost - optimum) / optimum
            gaps.append(gap)
            logger.info(f"{name}: {cost} (Gap: {gap:.4%})")
        logger.info(f"Mean gap: {sum(gaps) / len(gaps):.4%}")

        # Dump as csv
        csv_file = f"{exp_dir}/results.csv"
        with open(csv_file, "w") as f:
            f.write("name,real_cost\n")
            for name, cost in real_costs.items():
                f.write(f"{name},{cost}\n")
