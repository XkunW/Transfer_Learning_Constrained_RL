import numpy as np
import configparser
from ast import literal_eval
import matplotlib.pyplot as plt
import os
import logging

from four_rooms import Gridworld

LOGGER = logging.getLogger()
handler = logging.FileHandler('training.log')
LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

def read_config(config_path: str) -> dict:
    parser = configparser.RawConfigParser()
    parser.optionxform = str
    parser.read(config_path)
    config = {}
    for section in parser.sections():
        config[section] = {
            key: literal_eval(value) for key, value in parser.items(section)
        }
    return config


def generate_four_rooms(maze: list, seed_val: int) -> Gridworld:
    np.random.seed(seed_val)
    rewards = dict(
        zip(["1", "2", "3"], list(np.random.uniform(low=-1.0, high=1.0, size=3)))
    )
    np.random.seed(seed_val)
    obstacles = dict(
        zip(["A", "B", "C"], list(np.random.uniform(low=1.0, high=2.0, size=3)))
    )
    maze = np.array(maze)
    return Gridworld(maze, rewards, obstacles)


def plot_mean_var(
    data: list,
    model_names: list,
    n_samples: int,
    n_tasks: int,
    is_reward: bool,
    figure_name: str,
) -> None:
    # plot the task return
    ticksize = 14
    textsize = 18

    plt.rc("font", size=textsize)  # controls default text sizes
    plt.rc("axes", titlesize=textsize)  # fontsize of the axes title
    plt.rc("axes", labelsize=textsize)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=ticksize)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=ticksize)  # fontsize of the tick labels
    plt.rc("legend", fontsize=ticksize)  # legend fontsize

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    for i, name in enumerate(model_names):
        mean = data[i].mean
        n_sample_per_tick = n_samples * n_tasks // mean.size
        x = np.arange(mean.size) * n_sample_per_tick
        se = data[i].calculate_standard_error()
        plt.plot(x, mean, label=name)
        ax.fill_between(x, mean - se, mean + se, alpha=0.3)
    plt.xlabel("sample")
    y_label = "reward" if is_reward else "cost"
    plt.ylabel(f"cumulative {y_label}")
    plt.title(f"Cumulative Training {y_label.capitalize()} Per Task")
    plt.tight_layout()
    plt.legend(ncol=2, frameon=False)
    plt.savefig(os.path.join("figures", f"{figure_name}.png"))


class MeanVar:
    """Class for computing and updating mean and variance using Welford's online algorithm"""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0
        self.m2 = 0

    def update(self, new_value) -> None:
        new_value = np.array(new_value)
        self.count += 1
        delta = new_value - self.mean
        self.mean += delta / self.count
        delta2 = new_value - self.mean
        self.m2 += delta * delta2

    def compute_sample_variance(self) -> float:
        return self.m2 / (self.count - 1)

    def calculate_standard_error(self) -> float:
        return np.sqrt(self.compute_sample_variance() / self.count)
