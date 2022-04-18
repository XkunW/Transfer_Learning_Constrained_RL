"""
Main training loop for all reinforcement learning agents
"""
from src.agents.constrained_ql_agent import ConstrainedQL
from src.agents.trajectory_constrained_ql_agent_v1 import TrajectoryConstrainedQLV1
from src.agents.trajectory_constrained_ql_agent_v2 import TrajectoryConstrainedQLV2
from src.agents.succ_feature_ql_agent import SuccessorFeatureQL
from src.agents.safety_feature_ql_agent import SafetyFeatureQL
from src.agents.ql_agent import QL
from src.features.tabular_succ_features import TabularSuccessorFeatures
from src.features.tabular_safety_features import TabularSafetyFeatures
from src.utils import create_dirs, init_logger, read_config, generate_four_rooms, MeanVar, plot_mean_var


def main() -> None:
    """
    Main training loop for all reinforcement learning agents
    """
    # Initial setup
    create_dirs()
    logger = init_logger("root")
    training_params = read_config("training.cfg")
    env_params = read_config("four_rooms.cfg")
    agent_params = read_config("agents.cfg")
    feature_params = read_config("features.cfg")

    # Initialize agents
    constrained_ql = ConstrainedQL(**agent_params["constrained_ql"], **agent_params["agent"])
    trajectory_constrained_ql_v1 = TrajectoryConstrainedQLV1(**agent_params["trajectory_constrained_ql_v1"], **agent_params["agent"])
    trajectory_constrained_ql_v2 = TrajectoryConstrainedQLV2(**agent_params["trajectory_constrained_ql_v2"], **agent_params["agent"])
    succ_feature_ql = SuccessorFeatureQL(TabularSuccessorFeatures(**feature_params["tabular_sf"]), **agent_params["succ_feat_ql"], **agent_params["agent"])
    safe_feature_ql = SafetyFeatureQL(TabularSafetyFeatures(**feature_params["tabular_safety"]), **agent_params["safe_feat_ql"], **agent_params["agent"])
    ql = QL(**agent_params["ql"], **agent_params["agent"])
    agents = {
        "ConstrainedQL": constrained_ql, 
        "TrajectoryConstrainedQLV1": trajectory_constrained_ql_v1,
        "TrajectoryConstrainedQLV2": trajectory_constrained_ql_v2,
        "SuccessorFeatureQL": succ_feature_ql,
        "SafetyFeatureQL": safe_feature_ql,
        "QL": ql,
    }

    # Initialize performance statistics 
    task_return_hist = [MeanVar() for _ in agents]
    task_cost_hist = [MeanVar() for _ in agents]
    task_reward_collection_hist = [MeanVar() for _ in agents]

    constrained_agents = {name:a for (name, a) in agents.items() if hasattr(a, "threshold")}
    task_interval_violate_hist = [MeanVar() for _ in constrained_agents]
    task_constraint_violate_hist = [MeanVar() for _ in constrained_agents]

    n_trials = training_params["general"]["n_trials"]
    n_samples = training_params["general"]["n_samples"]
    n_tasks = training_params["general"]["n_tasks"]
    for trial in range(n_trials):
        for name in agents:
            agents[name].initialize()
        for i in range(n_tasks):
            task = generate_four_rooms(env_params["env"]["maze"], i)
            for name in agents:
                info_str = "trial {}, task {}, solving with {}".format(trial, i, name)
                print(info_str)
                logger.info("\n" + info_str)
                agents[name].train_on_task(task, n_samples)
        # Update stats
        for i, name in enumerate(agents):
            task_return_hist[i].update(agents[name].reward_hist)
            task_cost_hist[i].update(agents[name].cost_hist)
            task_reward_collection_hist[i].update(agents[name].reward_collection_hist)
        for i, name in enumerate(constrained_agents):
            task_interval_violate_hist[i].update(agents[name].interval_violate_hist)
            task_constraint_violate_hist[i].update(constrained_agents[name].violation_hist)
    
    plot_mean_var(task_return_hist, agents.keys(), n_samples, n_tasks, "reward", "return_comparison",)
    plot_mean_var(task_cost_hist, agents.keys(), n_samples, n_tasks, "cost", "cost_comparison",)
    plot_mean_var(task_interval_violate_hist, agents.keys(), n_samples, n_tasks, "interval violations", "episode_termination_comparison",)
    plot_mean_var(task_constraint_violate_hist, constrained_agents.keys(), n_samples, n_tasks, "violations", "violation_comparison",)
    plot_mean_var(task_reward_collection_hist, agents.keys(), n_samples, n_tasks, "reward collection", "reward_collection_comparison",)


if __name__ == "__main__":
    main()
