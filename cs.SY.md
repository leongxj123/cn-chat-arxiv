# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Automated Security Response through Online Learning with Adaptive Conjectures](https://arxiv.org/abs/2402.12499) | 该论文通过自适应猜想的在线学习，提出了一种适用于IT基础设施的自动化安全响应方法，其中游戏参与者通过Bayesian学习调整猜想，并通过推演更新策略，最终实现了最佳拟合，提高了推演在猜想模型下的性能。 |
| [^2] | [FIRE: A Failure-Adaptive Reinforcement Learning Framework for Edge Computing Migrations](https://arxiv.org/abs/2209.14399) | 提出了一个面向边缘计算迁移的故障自适应强化学习框架 FIRE，引入ImRE算法，通过在边缘计算数字孪生环境中训练RL策略来适应罕见事件，解决了RL框架在处理偶发服务器故障方面的挑战。 |
| [^3] | [Congestion Pricing for Efficiency and Equity: Theory and Applications to the San Francisco Bay Area.](http://arxiv.org/abs/2401.16844) | 本研究提出了一种新的拥堵定价方案，旨在同时减少交通拥堵水平和缩小不同旅行者之间的成本差异，从而提高效率和公平性。 |
| [^4] | [Predictive Analysis for Optimizing Port Operations.](http://arxiv.org/abs/2401.14498) | 本研究开发了一种具有竞争预测和分类能力的港口运营解决方案，用于准确估计船舶在港口的总时间和延迟时间，填补了港口分析模型在这方面的空白，并为海事物流领域提供了有价值的贡献。 |
| [^5] | [Model-Free Learning and Optimal Policy Design in Multi-Agent MDPs Under Probabilistic Agent Dropout.](http://arxiv.org/abs/2304.12458) | 本文研究了多智能体MDP中基于概率代理掉线的情况，并提出了一种无模型算法，能够消除掉线情况需要枚举计算的限制，从而实现计算后掉线系统的最优策略设计。 |
| [^6] | [D3G: Learning Multi-robot Coordination from Demonstrations.](http://arxiv.org/abs/2207.08892) | 本文提出了一个D3G框架，可以从演示中学习多机器人协调。通过最小化轨迹与演示之间的不匹配，每个机器人可以自动调整其个体动态和目标，提高了学习效率和效果。 |

# 详细

[^1]: 通过自适应猜想的在线学习实现自动化安全响应

    Automated Security Response through Online Learning with Adaptive Conjectures

    [https://arxiv.org/abs/2402.12499](https://arxiv.org/abs/2402.12499)

    该论文通过自适应猜想的在线学习，提出了一种适用于IT基础设施的自动化安全响应方法，其中游戏参与者通过Bayesian学习调整猜想，并通过推演更新策略，最终实现了最佳拟合，提高了推演在猜想模型下的性能。

    

    我们研究了针对IT基础设施的自动化安全响应，并将攻击者和防御者之间的互动形式表述为一个部分观测、非平稳博弈。我们放宽了游戏模型正确规定的标准假设，并考虑每个参与者对模型有一个概率性猜想，可能在某种意义上错误规定，即真实模型的概率为0。这种形式允许我们捕捉关于基础设施和参与者意图的不确定性。为了在线学习有效的游戏策略，我们设计了一种新颖的方法，其中一个参与者通过贝叶斯学习迭代地调整其猜想，并通过推演更新其策略。我们证明了猜想会收敛到最佳拟合，并提供了在具有猜测模型的情况下推演实现性能改进的上限。为了刻画游戏的稳定状态，我们提出了Berk-Nash平衡的一个变种。

    arXiv:2402.12499v1 Announce Type: cross  Abstract: We study automated security response for an IT infrastructure and formulate the interaction between an attacker and a defender as a partially observed, non-stationary game. We relax the standard assumption that the game model is correctly specified and consider that each player has a probabilistic conjecture about the model, which may be misspecified in the sense that the true model has probability 0. This formulation allows us to capture uncertainty about the infrastructure and the intents of the players. To learn effective game strategies online, we design a novel method where a player iteratively adapts its conjecture using Bayesian learning and updates its strategy through rollout. We prove that the conjectures converge to best fits, and we provide a bound on the performance improvement that rollout enables with a conjectured model. To characterize the steady state of the game, we propose a variant of the Berk-Nash equilibrium. We 
    
[^2]: FIRE：面向边缘计算迁移的故障自适应强化学习框架

    FIRE: A Failure-Adaptive Reinforcement Learning Framework for Edge Computing Migrations

    [https://arxiv.org/abs/2209.14399](https://arxiv.org/abs/2209.14399)

    提出了一个面向边缘计算迁移的故障自适应强化学习框架 FIRE，引入ImRE算法，通过在边缘计算数字孪生环境中训练RL策略来适应罕见事件，解决了RL框架在处理偶发服务器故障方面的挑战。

    

    在边缘计算中，用户服务配置文件由于用户移动而进行迁移。已经提出了强化学习（RL）框架来进行迁移，通常是在模拟数据上进行训练。然而，现有的RL框架忽视了偶发的服务器故障，尽管罕见，但会影响到像自动驾驶和实时障碍检测等对延迟敏感的应用。因此，这些（罕见事件）故障虽然在历史训练数据中没有得到充分代表，却对基于数据驱动的RL算法构成挑战。由于在实际应用中调整故障频率进行训练是不切实际的，我们引入了FIRE，这是一个通过在边缘计算数字孪生环境中训练RL策略来适应罕见事件的框架。我们提出了ImRE，一种基于重要性抽样的Q-learning算法，它根据罕见事件对值函数的影响进行比例抽样。FIRE考虑了延迟、迁移、故障和备份pl

    arXiv:2209.14399v2 Announce Type: replace-cross  Abstract: In edge computing, users' service profiles are migrated due to user mobility. Reinforcement learning (RL) frameworks have been proposed to do so, often trained on simulated data. However, existing RL frameworks overlook occasional server failures, which although rare, impact latency-sensitive applications like autonomous driving and real-time obstacle detection. Nevertheless, these failures (rare events), being not adequately represented in historical training data, pose a challenge for data-driven RL algorithms. As it is impractical to adjust failure frequency in real-world applications for training, we introduce FIRE, a framework that adapts to rare events by training a RL policy in an edge computing digital twin environment. We propose ImRE, an importance sampling-based Q-learning algorithm, which samples rare events proportionally to their impact on the value function. FIRE considers delay, migration, failure, and backup pl
    
[^3]: 用于效率和公平性的拥堵定价：理论及其在旧金山湾区的应用

    Congestion Pricing for Efficiency and Equity: Theory and Applications to the San Francisco Bay Area. (arXiv:2401.16844v1 [cs.GT])

    [http://arxiv.org/abs/2401.16844](http://arxiv.org/abs/2401.16844)

    本研究提出了一种新的拥堵定价方案，旨在同时减少交通拥堵水平和缩小不同旅行者之间的成本差异，从而提高效率和公平性。

    

    拥堵定价被许多城市用于缓解交通拥堵，但由于对低收入旅行者影响较大，引发了关于社会经济差距扩大的担忧。在本研究中，我们提出了一种新的拥堵定价方案，不仅可以最大限度地减少交通拥堵，还可以将公平性目标纳入其中，以减少不同支付意愿的旅行者之间的成本差异。我们的分析基于一个具有异质旅行者群体的拥堵博弈模型。我们提出了四种考虑实际因素的定价方案，例如对不同旅行者群体收取差异化的通行费以及征收整个路网中的所有边或只征收其中一部分边的选择。我们在旧金山湾区的校准高速公路网络中评估了我们的定价方案。我们证明了拥堵定价方案可以提高效率（即减少平均旅行时间）和公平性。

    Congestion pricing, while adopted by many cities to alleviate traffic congestion, raises concerns about widening socioeconomic disparities due to its disproportionate impact on low-income travelers. In this study, we address this concern by proposing a new class of congestion pricing schemes that not only minimize congestion levels but also incorporate an equity objective to reduce cost disparities among travelers with different willingness-to-pay. Our analysis builds on a congestion game model with heterogeneous traveler populations. We present four pricing schemes that account for practical considerations, such as the ability to charge differentiated tolls to various traveler populations and the option to toll all or only a subset of edges in the network. We evaluate our pricing schemes in the calibrated freeway network of the San Francisco Bay Area. We demonstrate that the proposed congestion pricing schemes improve both efficiency (in terms of reduced average travel time) and equit
    
[^4]: 优化港口运营的预测分析

    Predictive Analysis for Optimizing Port Operations. (arXiv:2401.14498v1 [cs.LG])

    [http://arxiv.org/abs/2401.14498](http://arxiv.org/abs/2401.14498)

    本研究开发了一种具有竞争预测和分类能力的港口运营解决方案，用于准确估计船舶在港口的总时间和延迟时间，填补了港口分析模型在这方面的空白，并为海事物流领域提供了有价值的贡献。

    

    海运是远距离和大宗货物运输的重要物流方式。然而，这种运输模式中复杂的规划经常受到不确定性的影响，包括天气条件、货物多样性和港口动态，导致成本增加。因此，准确估计船舶在港口停留的总时间和潜在延迟变得至关重要，以便在港口运营中进行有效的规划和安排。本研究旨在开发具有竞争预测和分类能力的港口运营解决方案，用于估计船舶的总时间和延迟时间。该研究填补了港口分析模型在船舶停留和延迟时间方面的重要空白，为海事物流领域提供了有价值的贡献。所提出的解决方案旨在协助港口环境下的决策制定，并预测服务延迟。通过对巴西港口的案例研究进行验证，同时使用特征分析来理解...

    Maritime transport is a pivotal logistics mode for the long-distance and bulk transportation of goods. However, the intricate planning involved in this mode is often hindered by uncertainties, including weather conditions, cargo diversity, and port dynamics, leading to increased costs. Consequently, accurately estimating vessel total (stay) time at port and potential delays becomes imperative for effective planning and scheduling in port operations. This study aims to develop a port operation solution with competitive prediction and classification capabilities for estimating vessel Total and Delay times. This research addresses a significant gap in port analysis models for vessel Stay and Delay times, offering a valuable contribution to the field of maritime logistics. The proposed solution is designed to assist decision-making in port environments and predict service delays. This is demonstrated through a case study on Brazil ports. Additionally, feature analysis is used to understand
    
[^5]: 多智能体MDP中基于概率代理掉线的无模型学习和最优策略设计

    Model-Free Learning and Optimal Policy Design in Multi-Agent MDPs Under Probabilistic Agent Dropout. (arXiv:2304.12458v1 [eess.SY])

    [http://arxiv.org/abs/2304.12458](http://arxiv.org/abs/2304.12458)

    本文研究了多智能体MDP中基于概率代理掉线的情况，并提出了一种无模型算法，能够消除掉线情况需要枚举计算的限制，从而实现计算后掉线系统的最优策略设计。

    

    本文研究了一个多智能体马尔可夫决策过程（MDP），该过程可以经历代理掉线，并基于对于策略的控制和预代理过程的采样来计算后掉线系统的策略。控制器的目标是寻找一个最优策略，使得在已知代理掉出概率的先验知识的情况下，期望系统的价值最大化。对于任何特定的掉线情况下的最优策略是这个问题的一个特例。对于具有特定转换独立性和奖励可分性结构的MDPs，我们假设从系统中移除代理组成了一个新的MDP，由剩余代理组成具有新状态和动作空间的MDP，转换动态消除已删除的代理，奖励与已删除的代理无关。首先我们展示了在这些假设下，对于预掉出系统期望值可以通过一个单一的MDP来表示；这个“鲁棒MDP”能够消除在计算最优策略时要评估所有$2^N$种代理掉线情况的需要。然后我们提出了一个无模型算法，该算法使用蒙特卡罗采样和重要性采样来学习鲁棒MDP，从而能够计算后掉线系统的最优策略。仿真结果展示了该方法的优点。

    This work studies a multi-agent Markov decision process (MDP) that can undergo agent dropout and the computation of policies for the post-dropout system based on control and sampling of the pre-dropout system. The controller's objective is to find an optimal policy that maximizes the value of the expected system given a priori knowledge of the agents' dropout probabilities. Finding an optimal policy for any specific dropout realization is a special case of this problem. For MDPs with a certain transition independence and reward separability structure, we assume that removing agents from the system forms a new MDP comprised of the remaining agents with new state and action spaces, transition dynamics that marginalize the removed agents, and rewards that are independent of the removed agents. We first show that under these assumptions, the value of the expected post-dropout system can be represented by a single MDP; this "robust MDP" eliminates the need to evaluate all $2^N$ realizations
    
[^6]: D3G: 从演示中学习多机器人协调

    D3G: Learning Multi-robot Coordination from Demonstrations. (arXiv:2207.08892v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2207.08892](http://arxiv.org/abs/2207.08892)

    本文提出了一个D3G框架，可以从演示中学习多机器人协调。通过最小化轨迹与演示之间的不匹配，每个机器人可以自动调整其个体动态和目标，提高了学习效率和效果。

    

    本文开发了一个分布式可微动态游戏（D3G）框架，可以实现从演示中学习多机器人协调。我们将多机器人协调表示为一个动态游戏，其中一个机器人的行为受其自身动态和目标的控制，同时也取决于其他机器人的行为。因此，通过调整每个机器人的目标和动态，可以适应协调。所提出的D3G使每个机器人通过最小化其轨迹与演示之间的不匹配，在分布式方式下自动调整其个体动态和目标。该学习框架具有新的设计，包括一个前向传递，所有机器人合作寻找游戏的纳什均衡，以及一个反向传递，在通信图中传播梯度。我们在仿真中测试了D3G，并给出了不同任务配置的两种机器人。结果证明了D3G学习多机器人协调的能力。

    This paper develops a Distributed Differentiable Dynamic Game (D3G) framework, which enables learning multi-robot coordination from demonstrations. We represent multi-robot coordination as a dynamic game, where the behavior of a robot is dictated by its own dynamics and objective that also depends on others' behavior. The coordination thus can be adapted by tuning the objective and dynamics of each robot. The proposed D3G enables each robot to automatically tune its individual dynamics and objectives in a distributed manner by minimizing the mismatch between its trajectory and demonstrations. This learning framework features a new design, including a forward-pass, where all robots collaboratively seek Nash equilibrium of a game, and a backward-pass, where gradients are propagated via the communication graph. We test the D3G in simulation with two types of robots given different task configurations. The results validate the capability of D3G for learning multi-robot coordination from de
    

