from agents.constrained_ql_agent import ConstrainedQL
from agents.trajectory_constrained_ql_agent import TrajectoryConstrainedQL
from agents.trajectory_constrained_ql_agent_v2 import TrajectoryConstrainedQLV2
from agents.sfql_agent import SFQL
from agents.ql_agent import QL
from successor_features.tabular_sf import TabularSF
import utils

# general training params
CONFIG_PARAMS = utils.read_config("constrained_sfql.cfg")


def main() -> None:
    # agents
    constrained_ql = ConstrainedQL(**CONFIG_PARAMS["constrained_ql"], **CONFIG_PARAMS["agent"])
    trajectory_constrained_ql = TrajectoryConstrainedQL(**CONFIG_PARAMS["trajectory_constrained_ql"], **CONFIG_PARAMS["agent"])
    trajectory_constrained_ql_v2 = TrajectoryConstrainedQLV2(**CONFIG_PARAMS["trajectory_constrained_ql_v2"], **CONFIG_PARAMS["agent"])
    sfql = SFQL(TabularSF(**CONFIG_PARAMS["tabular_sf"]), **CONFIG_PARAMS["sfql"], **CONFIG_PARAMS["agent"])
    ql = QL(**CONFIG_PARAMS["ql"], **CONFIG_PARAMS["agent"])
    agents = {
        "ConstrainedQL": constrained_ql, 
        "TrajectoryConstrainedQL": trajectory_constrained_ql,
        "TrajectoryConstrainedQLV2": trajectory_constrained_ql_v2,
        "SFQL": sfql,
        "QL": ql,
    }

    # train
    data_task_return = [utils.MeanVar() for _ in agents]
    data_task_cost = [utils.MeanVar() for _ in agents]
    n_trials = CONFIG_PARAMS["general"]["n_trials"]
    n_samples = CONFIG_PARAMS["general"]["n_samples"]
    n_tasks = CONFIG_PARAMS["general"]["n_tasks"]
    for trial in range(n_trials):
        # train each agent on a set of tasks
        for name in agents:
            agents[name].reset()
        for i in range(n_tasks):
            task = utils.generate_four_rooms(CONFIG_PARAMS["env"]["maze"], i)
            for name in agents:
                print("\ntrial {}, solving with {}".format(trial, name))
                agents[name].train_on_task(task, n_samples)

        # update performance statistics
        for i, name in enumerate(agents):
            data_task_return[i].update(agents[name].reward_hist)
            data_task_cost[i].update(agents[name].cost_hist)
    
    utils.plot_mean_var(data_task_return, agents.keys(), n_samples, n_tasks, True, "return_comparison")
    utils.plot_mean_var(data_task_cost, agents.keys(), n_samples, n_tasks, False, "cost_comparison")


if __name__ == "__main__":
    main()
