"""
Utility functions supporting training
"""
import os
import logging
import configparser
from ast import literal_eval
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.envs.four_rooms import FourRooms


def get_dir_path(dir_name: str) -> str:
    """
    Get the directory path

    Args:
        dir_name: Directory name

    Returns:
        str: Directory path
    """
    curr_file_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(curr_file_dir)
    dir_path = os.path.join(parent_dir, dir_name)
    return dir_path


def init_logger(name: str) -> logging.Logger:
    """
    Initializes logger object

    Args:
        name: Logger name

    Returns:
        logging.Logger: Global Logger object
    """
    log_dir = get_dir_path("log")
    log_file_path = os.path.join(log_dir, "training.log")
    log_file = Path(log_file_path)
    log_file.touch(exist_ok=True)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel("INFO")
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)

    return logger


def create_dirs() -> None:
    """
    Create the necessary directories if they do not exist
    """
    dir_names = ["log", "figures"]
    for dir_name in dir_names:
        dir_path = get_dir_path(dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def read_config(config_name: str) -> dict:
    """
    Reads config file

    Args:
        config_name (str): Config file name

    Returns:
        dict: Dictionary of config
    """
    config_dir_path = get_dir_path("configs")
    config_path = os.path.join(config_dir_path, config_name)

    parser = configparser.RawConfigParser()
    parser.optionxform = str
    parser.read(config_path)
    config = {}
    for section in parser.sections():
        config[section] = {
            key: literal_eval(value) for key, value in parser.items(section)
        }
    return config


def generate_four_rooms(maze: list, seed_val: int) -> FourRooms:
    """
    Generates the four rooms environment

    Args:
        maze (list): Environment setup
        seed_val (int): Seed value for randomizing rewards and costs

    Returns:
        FourRooms: The newly generated environment
    """
    np.random.seed(seed_val)
    rewards = dict(
        zip(["1", "2", "3"], list(np.random.uniform(low=0.0, high=2.0, size=3)))
    )
    np.random.seed(seed_val)
    obstacles = dict(
        zip(["A", "B", "C"], list(np.random.uniform(low=1.0, high=2.0, size=3)))
    )
    maze = np.array(maze)
    return FourRooms(maze, rewards, obstacles)


def plot_mean_var(
    data: list,
    model_names: list,
    n_samples: int,
    n_tasks: int,
    y_label: str,
    figure_name: str,
    save_fig: bool = True,
    show_fig: bool = False
) -> None:
    """
    Plots mean and variance

    Args:
        data (list): Source data
        model_names (list): Agent names
        n_samples (int): Number of samples
        n_tasks (int): Number of tasks
        y_label (str): Y axis label
        figure_name (str): Figure name
        save_fig (bool, optional): Whether to save figure. Defaults to True.
        show_fig (bool, optional): Whether to display figure. Defaults to False.
    """
    ticksize = 14
    textsize = 18

    plt.rc("font", size=textsize)  
    plt.rc("axes", titlesize=textsize)  
    plt.rc("axes", labelsize=textsize)  
    plt.rc("xtick", labelsize=ticksize) 
    plt.rc("ytick", labelsize=ticksize) 
    plt.rc("legend", fontsize=ticksize) 

    plt.figure(figsize=(15, 6))
    ax = plt.gca()
    for i, name in enumerate(model_names):
        mean = data[i].mean
        n_sample_per_tick = n_samples * n_tasks // mean.size
        x = np.arange(mean.size) * n_sample_per_tick
        se = data[i].calculate_standard_error()
        plt.plot(x, mean, label=name)
        ax.fill_between(x, mean - se, mean + se, alpha=0.3)
    plt.xlabel("sample")
    plt.ylabel(f"cumulative {y_label}")
    plt.title(f"Cumulative Training {y_label.capitalize()} Per Task")
    plt.tight_layout()
    plt.legend(ncol=2, frameon=False)
    if save_fig:
        plt.savefig(os.path.join("figures", f"{figure_name}.png"))
    if show_fig:
        plt.show()


class MeanVar:
    """
    Class for computing and updating mean and variance using Welford's online algorithm
    """
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
