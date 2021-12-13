# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="optuna-hydra-sweeper")
def sphere(cfg: DictConfig) -> float:
    x: float = cfg.x
    y: float = cfg.y
    print(cfg)
    return x ** 2 + y ** 2


if __name__ == "__main__":
    print(sphere())