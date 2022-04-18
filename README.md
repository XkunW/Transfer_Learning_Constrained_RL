# Transfer Learning for Constrained Reinforcement Learning

This repository introduces a novel **Transfer Learning algorithm for Constrained Reinforcement Learning** where both reward knowledege and cost knowledge are transferred between tasks. The algorithm is inspired by **Successor Features** and **Generalized Policy Improvement** from the "[Successor Features for Transfer in Reinforcement Learning](https://arxiv.org/pdf/1606.05312.pdf)" paper. The key idea is to have a second set of successor features, namely **Safety Features**, to learn the successor representations of the obstacles, so that the agent would try to keep the cost under the threshold while achieving maximum rewards. 

# Repository Organization

The `configs` directory contains all the training related paramter values, which are organized as follows:
* `agents.cfg`: Generic agent parameters and agent specific training parameters.
* `features.cfg`: The training parameters for successor features and safety features.
* `four_rooms.cfg`: The setup for the four rooms grid environment.
* `training.cfg`: The general training parameters.

The `src` directory contains all the reinforcement learning solution files, which are organized as follows:
* `agents`: The sub-directory for all the reinforcement agent code:
  * `agents.py`: The parent class for all agents.
  * `constrained_ql_agent.py`: Constrained Q Learning agent from the "[Deep Constrained Q Learning](https://arxiv.org/pdf/2003.09398.pdf)" paper. The threshold is set for the immeadiate cost. Thus an action is deemed unsafe if the immediate cost exceeds the threshold.
  * `ql_agent.py`: Q Learning agent.
  * `safety_feature_ql_agent.py`: Safety Feature Q Learning agent. The threshold is set for the entire trajectory.
  * `succ_feature_ql_agent.py`: Successor Feature Q Learning agent from "Successor Features for Transfer in Reinforcement Learning".
  * `trajectory_constrained_ql_agent_v1.py`: Trajectory Constrained Q Learning Agent inspired by Constrained Q Learning. The threshold is set for the entire trajectory. Thus an action is considered as unsafe if the sum of cumulative cost and immediate cost exceeds the threshold.
  * `trajectory_constrained_ql_agent_v2.py`: The second version of Trajectory Constrained Q Learning. The difference is an action is considered as unsafe if the sum of cumulative cost, and the current action cost J(s,a) exceeds the threshold.
  
* `envs`: The sub-directory for all the environment code:
  * `four_rooms.py`: A four rooms grid environment similar to the experiment setup from the "Successor Features for Transfer in Reinforcement Learning" paper.

* `features`: The sub-directory for all the successor features code:
  * `tabular_safety_features.py`: Inherited from the tabular successor features class with additional parameters and methods to compute and update safety features.
  * `tabular_succ_features.py`: Computes and updates tabular successor features from the "Successor Features for Transfer in Reinforcement Learning" paper.

* `util.py`: Utility functions supporting training.

The `main.py` contains the main training loop for the reinforcement learning agent, which generates a `log` directory for training logs and `figures` directory for training plots.

# Dependencies

* Python ≥ 3.9
* [NumPy](https://github.com/numpy/numpy)

# Reinforcement Learning Algorithm Breakdown

## State-Action Space

The current four rooms setup contains 3 types of rewards and 3 types of obstacles, rewards can only be collected once while obstacles can be hit multiple times. The agent starts from the bottom left corner and the goal is to reach the top right corner. There are walls that divides the grid into 4 rooms. The action space is **["up", "down", "left", "right"]**.

## Safety Feature Q Learning

Below is the pseudo code for the Safety Feature Q Learning algorithm

### Algorithm
#### **Safety Features**
$c(s, a, s') = {\phi}_c(s, a, s')^{\intercal} w_c\\$ 
$
\begin{aligned}
J^{\pi}(s, a) &= E^{\pi} \left[c_{t+1} + {\gamma} c_{t+2} + ...| S_t = s, A_t = a\right] \\
              &= E^{\pi} \left[{\phi}_{c_{t+1}}^\intercal w_c + {\gamma} {\phi}_{c_{t+2}}^\intercal w_c + ...| S_t = s, A_t = a\right] \\
              &= E^{\pi} \left[{\sum}_{i=t}^{\infty} {\gamma}^{i-t}{\phi}_{c_{i+1}} | S_t = s, A_t = a\right]^{\intercal} w_c \\
              &= {\psi}_c^{\pi}(s, a)^{\intercal} w_c \\
\end{aligned}
\$
${\psi}_c^{\pi}(s, a) = {\phi}_{c_{t+1}} + {\gamma}E^{\pi} \left[{\psi}_c^{\pi}(S_{t+1}, {\pi}(S_{t+1}))|S_t = s, A_t = a\right]\\$
\
#### **Safe Generalized Policy Improvement (GPI)**
$
Q^{All \pi}(s) = {\psi}^{All \pi}(s)^\intercal * w'\\ 
J^{All \pi}(s) = {\psi}_c^{All \pi}(s)^\intercal * w'_c\\
\
$
$Q^{{\pi}_{safe}}(s) =$ Assign negative $Q$ values to all ${{\pi}_i}(a)$ where $J^{{\pi}_i}(s, a)$ + current cost < threshold $\\$
${\pi_{src}} = argmax_{\pi_i}(max_a(Q^{{\pi}_{safe}}(s)))\\$

return $Q^{\pi_{src}}(s)$, $J^{\pi_{src}}(s)$

#### **Safe Generalized Policy Evaluation (GPE)**
$
Q^{\pi_{curr}}(s) = {\psi}^{\pi_{curr}}(s)^\intercal * w'\\ 
J^{\pi_{curr}}(s) = {\psi}_c^{\pi_{curr}}(s)^\intercal * w'_c\\
\
$
$Q_{safe}(s) =$ Assign negative $Q$ values to all actions $a$ where $J^{{\pi}_curr}(s, a)$ + current cost < threshold $\\$
$a' = argmax(Q_{safe}(s))\\$
return $a'\\$

#### **Training**
For each sample:\
&emsp; $Q^{\pi_{src}}(s)$, $J^{\pi_{src}}(s)$ = Safe GPI($s$)\
&emsp; $a$ = Epsilon Greedy($Q^{\pi_{src}}(s)$)\
&emsp; $s', r, c$ = State Transition($s, a$)\
&emsp; Update Reward Weight($\phi$, $r$, current task index)\
&emsp; Update Cost Weight($\phi_c$, $c$, current task index)\
&emsp; $Q^{\pi_{src'}}(s')$, $J^{\pi_{src'}}(s')$ = Safe GPI($s'$)\
&emsp; $a'$ = $argmax(Q^{\pi_{src'}}(s'))$\
&emsp; Update Successor Features($s, a, \phi, s', a', \gamma$, current_task_index)\
&emsp; Update Safety Features($s, a, \phi_c, s', a', \gamma$, current_task_index)\
&emsp; If current_task_index != $src$:\
&emsp;&emsp; $a'$ = Safe GPE($s', src$)\
&emsp;&emsp; Update Successor Features($s, a, \phi, s', a', \gamma, src$)\
&emsp;&emsp; Update Safety Features($s, a, \phi_c, s', a', \gamma, src$)