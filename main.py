from agents.constrained_ql_agent import ConstrainedQL
from agents.sfql_agent import SFQL
from agents.ql_agent import QL
from successor_features.tabular_sf import TabularSF
import utils

# general training params
CONFIG_PARAMS = utils.read_config("constrained_sfql.cfg")


def main() -> None:
    # agents
    constrained_ql = ConstrainedQL(**CONFIG_PARAMS["constrained_ql"], **CONFIG_PARAMS["agent"])
    sfql = SFQL(TabularSF(**CONFIG_PARAMS["tabular_sf"]), **CONFIG_PARAMS["agent"])
    ql = QL(**CONFIG_PARAMS["ql"], **CONFIG_PARAMS["agent"])
    agents = [constrained_ql, sfql, ql]
    agents = [constrained_ql, ql]
    names = ["ConstrainedQL", "SFQL", "QL"]
    names = ["ConstrainedQL", "QL"]

    # train
    data_task_return = [utils.MeanVar() for _ in agents]
    data_task_cost = [utils.MeanVar() for _ in agents]
    n_trials = CONFIG_PARAMS["general"]["n_trials"]
    n_samples = CONFIG_PARAMS["general"]["n_samples"]
    n_tasks = CONFIG_PARAMS["general"]["n_tasks"]
    for trial in range(n_trials):
        # train each agent on a set of tasks
        for agent in agents:
            agent.reset()
        for _ in range(n_tasks):
            task = utils.generate_four_rooms(CONFIG_PARAMS["env"]["maze"])
            for agent, name in zip(agents, names):
                print("\ntrial {}, solving with {}".format(trial, name))
                agent.train_on_task(task, n_samples)

        # update performance statistics
        for i, agent in enumerate(agents):
            data_task_return[i].update(agent.reward_hist)
            data_task_cost[i].update(agent.cost_hist)
    
    utils.plot_mean_var(data_task_return, names, n_samples, n_tasks, True, "return_comparison")
    utils.plot_mean_var(data_task_cost, names, n_samples, n_tasks, False, "cost_comparison")


if __name__ == "__main__":
    main()
