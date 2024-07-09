# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Incentive-Compatible Vertiport Reservation in Advanced Air Mobility: An Auction-Based Approach](https://arxiv.org/abs/2403.18166) | 本论文提出了一种激励兼容且个体理性的vertiport预订机制，旨在协调多个运营商操作的电动垂直起降（eVTOL）飞行器在vertiports之间移动，最大化所有运营商的总估值同时最小化vertiports的拥堵。 |
| [^2] | [Blending Data-Driven Priors in Dynamic Games](https://arxiv.org/abs/2402.14174) | 探索一种在动态游戏中将数据驱动参考政策与基于优化博弈政策相融合的方法，提出了一种非合作动态博弈KLGame，其中包含了针对每个决策者的可调参数。 |
| [^3] | [An active learning method for solving competitive multi-agent decision-making and control problems.](http://arxiv.org/abs/2212.12561) | 我们提出了一个基于主动学习的方法，用于解决竞争性多智能体决策和控制问题。通过重构私有策略和预测稳态行动配置文件，外部观察者可以成功进行预测和优化策略。 |
| [^4] | [Towards Improving Operation Economics: A Bilevel MIP-Based Closed-Loop Predict-and-Optimize Framework for Prescribing Unit Commitment.](http://arxiv.org/abs/2208.13065) | 本文提出了一个基于双层 MIP 的闭环预测优化框架，使用成本导向的预测器来改进电力系统的经济运行。该框架通过反馈循环迭代地改进预测器，实现了对机组组合的最佳操作。 |

# 详细

[^1]: 具有激励兼容性的先进空中移动中的Vertiport预订：基于拍卖的方法

    Incentive-Compatible Vertiport Reservation in Advanced Air Mobility: An Auction-Based Approach

    [https://arxiv.org/abs/2403.18166](https://arxiv.org/abs/2403.18166)

    本论文提出了一种激励兼容且个体理性的vertiport预订机制，旨在协调多个运营商操作的电动垂直起降（eVTOL）飞行器在vertiports之间移动，最大化所有运营商的总估值同时最小化vertiports的拥堵。

    

    预计未来先进空中移动（AAM）的崛起将成为一个数十亿美元的产业。市场机制被认为是AAM运营的一个重要组成部分，其中包括具有私人估值的异构运营商。本文研究了设计一种机制来协调由多个具有与其机群关联的异构估值的运营商操作的电动垂直起降（eVTOL）飞行器在vertiports之间移动的问题，同时强制执行vertiports的到达、离开和停车约束。特别是，我们提出了一种激励兼容且个体理性的vertiport预订机制，该机制最大化了一个社会福利度量标准，从而包括最大化所有运营商的总估值同时最小化vertiports的拥堵。此外，我们提高了设计这一机制的计算可处理性

    arXiv:2403.18166v1 Announce Type: cross  Abstract: The rise of advanced air mobility (AAM) is expected to become a multibillion-dollar industry in the near future. Market-based mechanisms are touted to be an integral part of AAM operations, which comprise heterogeneous operators with private valuations. In this work, we study the problem of designing a mechanism to coordinate the movement of electric vertical take-off and landing (eVTOL) aircraft, operated by multiple operators each having heterogeneous valuations associated with their fleet, between vertiports, while enforcing the arrival, departure, and parking constraints at vertiports. Particularly, we propose an incentive-compatible and individually rational vertiport reservation mechanism that maximizes a social welfare metric, which encapsulates the objective of maximizing the overall valuations of all operators while minimizing the congestion at vertiports. Additionally, we improve the computational tractability of designing th
    
[^2]: 在动态游戏中融合数据驱动的先验知识

    Blending Data-Driven Priors in Dynamic Games

    [https://arxiv.org/abs/2402.14174](https://arxiv.org/abs/2402.14174)

    探索一种在动态游戏中将数据驱动参考政策与基于优化博弈政策相融合的方法，提出了一种非合作动态博弈KLGame，其中包含了针对每个决策者的可调参数。

    

    随着智能机器人如自动驾驶车辆在人群中的部署越来越多，这些系统应该在安全的、与人互动意识相关的运动规划中利用基于模型的博弈论规划器与数据驱动政策的程度仍然是一个悬而未决的问题。本文探讨了一种融合数据驱动参考政策和基于优化的博弈论政策的原则性方法。我们制定了KLGame，这是一种带有Kullback-Leibler（KL）正则化的非合作动态博弈，针对一个一般的、随机的，可能是多模式的参考政策。

    arXiv:2402.14174v1 Announce Type: cross  Abstract: As intelligent robots like autonomous vehicles become increasingly deployed in the presence of people, the extent to which these systems should leverage model-based game-theoretic planners versus data-driven policies for safe, interaction-aware motion planning remains an open question. Existing dynamic game formulations assume all agents are task-driven and behave optimally. However, in reality, humans tend to deviate from the decisions prescribed by these models, and their behavior is better approximated under a noisy-rational paradigm. In this work, we investigate a principled methodology to blend a data-driven reference policy with an optimization-based game-theoretic policy. We formulate KLGame, a type of non-cooperative dynamic game with Kullback-Leibler (KL) regularization with respect to a general, stochastic, and possibly multi-modal reference policy. Our method incorporates, for each decision maker, a tunable parameter that pe
    
[^3]: 解决竞争性多智能体决策和控制问题的主动学习方法

    An active learning method for solving competitive multi-agent decision-making and control problems. (arXiv:2212.12561v2 [eess.SY] UPDATED)

    [http://arxiv.org/abs/2212.12561](http://arxiv.org/abs/2212.12561)

    我们提出了一个基于主动学习的方法，用于解决竞争性多智能体决策和控制问题。通过重构私有策略和预测稳态行动配置文件，外部观察者可以成功进行预测和优化策略。

    

    我们提出了一种基于主动学习的方案，用于重构由相互作用代理人群体执行的私有策略，并预测底层多智能体交互过程的确切结果，这里被认为是一个稳定的行动配置文件。我们设想了一个场景，在这个场景中，一个具有学习程序的外部观察者可以通过私有的行动-反应映射进行查询和观察代理人的反应，集体的不动点对应于一个稳态配置文件。通过迭代地收集有意义的数据和更新行动-反应映射的参数估计，我们建立了评估所提出的主动学习方法的渐近性质的充分条件，以便如果收敛发生，它只能朝向一个稳态行动配置文件。这一事实导致了两个主要结果：i）学习局部精确的行动-反应映射替代物使得外部观察者能够成功完成其预测任务，ii）与代理人的互动提供了一种方法来优化策略以达到最佳效果。

    We propose a scheme based on active learning to reconstruct private strategies executed by a population of interacting agents and predict an exact outcome of the underlying multi-agent interaction process, here identified as a stationary action profile. We envision a scenario where an external observer, endowed with a learning procedure, can make queries and observe the agents' reactions through private action-reaction mappings, whose collective fixed point corresponds to a stationary profile. By iteratively collecting sensible data and updating parametric estimates of the action-reaction mappings, we establish sufficient conditions to assess the asymptotic properties of the proposed active learning methodology so that, if convergence happens, it can only be towards a stationary action profile. This fact yields two main consequences: i) learning locally-exact surrogates of the action-reaction mappings allows the external observer to succeed in its prediction task, and ii) working with 
    
[^4]: 改善运营经济学：基于双层 MIP 的闭环预测优化框架来预测机组组合的操作计划

    Towards Improving Operation Economics: A Bilevel MIP-Based Closed-Loop Predict-and-Optimize Framework for Prescribing Unit Commitment. (arXiv:2208.13065v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2208.13065](http://arxiv.org/abs/2208.13065)

    本文提出了一个基于双层 MIP 的闭环预测优化框架，使用成本导向的预测器来改进电力系统的经济运行。该框架通过反馈循环迭代地改进预测器，实现了对机组组合的最佳操作。

    

    通常，系统操作员在开环预测优化过程中进行电力系统的经济运行：首先预测可再生能源(RES)的可用性和系统储备需求；根据这些预测，系统操作员解决诸如机组组合(UC)的优化模型，以确定相应的经济运行计划。然而，这种开环过程可能会实质性地损害操作经济性，因为它的预测器目光短浅地寻求改善即时的统计预测误差，而不是最终的操作成本。为此，本文提出了一个闭环预测优化框架，提供一种预测机组组合以改善操作经济性的方法。首先，利用双层混合整数规划模型针对最佳系统操作训练成本导向的预测器。上层基于其引起的操作成本来训练 RES 和储备预测器；下层则在给定预测的 RES 和储备的情况下，依据最佳操作原则求解 UC。这两个层级通过反馈环路进行交互性互动，直到收敛为止。在修改后的IEEE 24-bus系统上的数值实验表明，与三种最先进的 UC 基准线相比，所提出的框架具有高效性和有效性。

    Generally, system operators conduct the economic operation of power systems in an open-loop predict-then-optimize process: the renewable energy source (RES) availability and system reserve requirements are first predicted; given the predictions, system operators solve optimization models such as unit commitment (UC) to determine the economical operation plans accordingly. However, such an open-loop process could essentially compromise the operation economics because its predictors myopically seek to improve the immediate statistical prediction errors instead of the ultimate operation cost. To this end, this paper presents a closed-loop predict-and-optimize framework, offering a prescriptive UC to improve the operation economics. First, a bilevel mixed-integer programming model is leveraged to train cost-oriented predictors tailored for optimal system operations: the upper level trains the RES and reserve predictors based on their induced operation cost; the lower level, with given pred
    

