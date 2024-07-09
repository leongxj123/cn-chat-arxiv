# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Incentive-Compatible Vertiport Reservation in Advanced Air Mobility: An Auction-Based Approach](https://arxiv.org/abs/2403.18166) | 本论文提出了一种激励兼容且个体理性的vertiport预订机制，旨在协调多个运营商操作的电动垂直起降（eVTOL）飞行器在vertiports之间移动，最大化所有运营商的总估值同时最小化vertiports的拥堵。 |
| [^2] | [Open Ad Hoc Teamwork with Cooperative Game Theory](https://arxiv.org/abs/2402.15259) | 提出了采用合作博弈论解释开放式即兴团队合作中联合Q值表示的新理论，为进一步发展这一研究方向和应用提供了新思路 |
| [^3] | [An active learning method for solving competitive multi-agent decision-making and control problems.](http://arxiv.org/abs/2212.12561) | 我们提出了一个基于主动学习的方法，用于解决竞争性多智能体决策和控制问题。通过重构私有策略和预测稳态行动配置文件，外部观察者可以成功进行预测和优化策略。 |

# 详细

[^1]: 具有激励兼容性的先进空中移动中的Vertiport预订：基于拍卖的方法

    Incentive-Compatible Vertiport Reservation in Advanced Air Mobility: An Auction-Based Approach

    [https://arxiv.org/abs/2403.18166](https://arxiv.org/abs/2403.18166)

    本论文提出了一种激励兼容且个体理性的vertiport预订机制，旨在协调多个运营商操作的电动垂直起降（eVTOL）飞行器在vertiports之间移动，最大化所有运营商的总估值同时最小化vertiports的拥堵。

    

    预计未来先进空中移动（AAM）的崛起将成为一个数十亿美元的产业。市场机制被认为是AAM运营的一个重要组成部分，其中包括具有私人估值的异构运营商。本文研究了设计一种机制来协调由多个具有与其机群关联的异构估值的运营商操作的电动垂直起降（eVTOL）飞行器在vertiports之间移动的问题，同时强制执行vertiports的到达、离开和停车约束。特别是，我们提出了一种激励兼容且个体理性的vertiport预订机制，该机制最大化了一个社会福利度量标准，从而包括最大化所有运营商的总估值同时最小化vertiports的拥堵。此外，我们提高了设计这一机制的计算可处理性

    arXiv:2403.18166v1 Announce Type: cross  Abstract: The rise of advanced air mobility (AAM) is expected to become a multibillion-dollar industry in the near future. Market-based mechanisms are touted to be an integral part of AAM operations, which comprise heterogeneous operators with private valuations. In this work, we study the problem of designing a mechanism to coordinate the movement of electric vertical take-off and landing (eVTOL) aircraft, operated by multiple operators each having heterogeneous valuations associated with their fleet, between vertiports, while enforcing the arrival, departure, and parking constraints at vertiports. Particularly, we propose an incentive-compatible and individually rational vertiport reservation mechanism that maximizes a social welfare metric, which encapsulates the objective of maximizing the overall valuations of all operators while minimizing the congestion at vertiports. Additionally, we improve the computational tractability of designing th
    
[^2]: 采用合作博弈论的开放式即兴团队合作

    Open Ad Hoc Teamwork with Cooperative Game Theory

    [https://arxiv.org/abs/2402.15259](https://arxiv.org/abs/2402.15259)

    提出了采用合作博弈论解释开放式即兴团队合作中联合Q值表示的新理论，为进一步发展这一研究方向和应用提供了新思路

    

    即兴团队合作面临着一个具有挑战性的问题，需要设计一个能够与队友协作但没有先前协调或联合训练的智能体。开放式即兴团队合作进一步复杂化了这一挑战，考虑了具有不断变化的队友数量的环境，即开放式团队。现有解决这一问题的最先进方法是基于图神经网络的策略学习（GPL），利用了图神经网络的泛化能力来处理无限数量的智能体，有效应对开放式团队。GPL的性能优于其他方法，但其联合Q值表示对解释造成了挑战，阻碍了进一步发展这一研究方向和应用。本文建立了一种新的理论，从合作博弈论的角度为GPL中采用的联合Q值表示提供了一种解释。基于我们的理论，我们提出了一种基于

    arXiv:2402.15259v1 Announce Type: cross  Abstract: Ad hoc teamwork poses a challenging problem, requiring the design of an agent to collaborate with teammates without prior coordination or joint training. Open ad hoc teamwork further complicates this challenge by considering environments with a changing number of teammates, referred to as open teams. The state-of-the-art solution to this problem is graph-based policy learning (GPL), leveraging the generalizability of graph neural networks to handle an unrestricted number of agents and effectively address open teams. GPL's performance is superior to other methods, but its joint Q-value representation presents challenges for interpretation, hindering further development of this research line and applicability. In this paper, we establish a new theory to give an interpretation for the joint Q-value representation employed in GPL, from the perspective of cooperative game theory. Building on our theory, we propose a novel algorithm based on
    
[^3]: 解决竞争性多智能体决策和控制问题的主动学习方法

    An active learning method for solving competitive multi-agent decision-making and control problems. (arXiv:2212.12561v2 [eess.SY] UPDATED)

    [http://arxiv.org/abs/2212.12561](http://arxiv.org/abs/2212.12561)

    我们提出了一个基于主动学习的方法，用于解决竞争性多智能体决策和控制问题。通过重构私有策略和预测稳态行动配置文件，外部观察者可以成功进行预测和优化策略。

    

    我们提出了一种基于主动学习的方案，用于重构由相互作用代理人群体执行的私有策略，并预测底层多智能体交互过程的确切结果，这里被认为是一个稳定的行动配置文件。我们设想了一个场景，在这个场景中，一个具有学习程序的外部观察者可以通过私有的行动-反应映射进行查询和观察代理人的反应，集体的不动点对应于一个稳态配置文件。通过迭代地收集有意义的数据和更新行动-反应映射的参数估计，我们建立了评估所提出的主动学习方法的渐近性质的充分条件，以便如果收敛发生，它只能朝向一个稳态行动配置文件。这一事实导致了两个主要结果：i）学习局部精确的行动-反应映射替代物使得外部观察者能够成功完成其预测任务，ii）与代理人的互动提供了一种方法来优化策略以达到最佳效果。

    We propose a scheme based on active learning to reconstruct private strategies executed by a population of interacting agents and predict an exact outcome of the underlying multi-agent interaction process, here identified as a stationary action profile. We envision a scenario where an external observer, endowed with a learning procedure, can make queries and observe the agents' reactions through private action-reaction mappings, whose collective fixed point corresponds to a stationary profile. By iteratively collecting sensible data and updating parametric estimates of the action-reaction mappings, we establish sufficient conditions to assess the asymptotic properties of the proposed active learning methodology so that, if convergence happens, it can only be towards a stationary action profile. This fact yields two main consequences: i) learning locally-exact surrogates of the action-reaction mappings allows the external observer to succeed in its prediction task, and ii) working with 
    

