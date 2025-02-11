# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Independent RL for Cooperative-Competitive Agents: A Mean-Field Perspective](https://arxiv.org/abs/2403.11345) | 本文从均场视角研究了独立强化学习在合作竞争代理中的应用，提出了一种可实现纳什均衡的线性二次结构RL方法，并通过考虑无限代理数量的情况来解决有限人口环境中的非稳态性问题。 |
| [^2] | [Incentive Allocation in Vertical Federated Learning Based on Bankruptcy Problem.](http://arxiv.org/abs/2307.03515) | 本文提出了一种基于破产问题的方法来解决垂直联邦学习中激励分配的挑战，以确保公平性和稳定性。 |
| [^3] | [Incentivizing Honesty among Competitors in Collaborative Learning and Optimization.](http://arxiv.org/abs/2305.16272) | 这项研究提出了一个模型来描述在协作学习中竞争对手的不诚实行为，提出了机制来激励诚实沟通，并确保学习质量与全面合作相当。 |

# 详细

[^1]: 独立强化学习用于合作竞争Agent：均场视角

    Independent RL for Cooperative-Competitive Agents: A Mean-Field Perspective

    [https://arxiv.org/abs/2403.11345](https://arxiv.org/abs/2403.11345)

    本文从均场视角研究了独立强化学习在合作竞争代理中的应用，提出了一种可实现纳什均衡的线性二次结构RL方法，并通过考虑无限代理数量的情况来解决有限人口环境中的非稳态性问题。

    

    在本文中，我们研究了分成团队的代理之间的强化学习（RL），每个团队内部存在合作，但不同团队之间存在非零和的竞争。为了开发一种可以明确实现纳什均衡的RL方法，我们专注于线性二次结构。此外，为了解决有限人口环境中由多智能体交互引起的非稳态性，我们考虑每个团队内代理数量无限的情况，即均场设置。这导致了一个广义和的LQ均场类型博弈（GS-MFTGs）。我们在标准逆可逆条件下表征了GS-MFTG的纳什均衡（NE）。然后证明了这个MFTG NE在有限人口博弈中为$\mathcal{O}(1/M)$-NE，其中$M$是每个团队中代理数量的下界。这些结构性结果推动了一个名为多玩家递进式自然Pol的算法。

    arXiv:2403.11345v1 Announce Type: cross  Abstract: We address in this paper Reinforcement Learning (RL) among agents that are grouped into teams such that there is cooperation within each team but general-sum (non-zero sum) competition across different teams. To develop an RL method that provably achieves a Nash equilibrium, we focus on a linear-quadratic structure. Moreover, to tackle the non-stationarity induced by multi-agent interactions in the finite population setting, we consider the case where the number of agents within each team is infinite, i.e., the mean-field setting. This results in a General-Sum LQ Mean-Field Type Game (GS-MFTGs). We characterize the Nash equilibrium (NE) of the GS-MFTG, under a standard invertibility condition. This MFTG NE is then shown to be $\mathcal{O}(1/M)$-NE for the finite population game where $M$ is a lower bound on the number of agents in each team. These structural results motivate an algorithm called Multi-player Receding-horizon Natural Pol
    
[^2]: 基于破产问题的垂直联邦学习中的激励分配

    Incentive Allocation in Vertical Federated Learning Based on Bankruptcy Problem. (arXiv:2307.03515v1 [cs.LG])

    [http://arxiv.org/abs/2307.03515](http://arxiv.org/abs/2307.03515)

    本文提出了一种基于破产问题的方法来解决垂直联邦学习中激励分配的挑战，以确保公平性和稳定性。

    

    垂直联邦学习（VFL）是一种有前景的方法，用于合作训练在不同参与方之间垂直划分的私有数据的机器学习模型。在VFL设置中，理想情况下，主动方（拥有带标签样本特征的参与方）通过与某些被动方（拥有相同样本但没有标签的额外特征的参与方）合作，在保护隐私的情况下改进其机器学习模型。然而，激励被动方参与VFL可能具有挑战性。本文重点研究了基于被动方在VFL过程中的贡献来为他们分配激励的问题。我们将这个问题定义为核心游戏论概念的一种变体——破产问题，并使用塔木德划分规则来解决它。我们在合成和真实数据集上评估了我们提出的方法，并展示它确保了激励的公平性和稳定性。

    Vertical federated learning (VFL) is a promising approach for collaboratively training machine learning models using private data partitioned vertically across different parties. Ideally in a VFL setting, the active party (party possessing features of samples with labels) benefits by improving its machine learning model through collaboration with some passive parties (parties possessing additional features of the same samples without labels) in a privacy preserving manner. However, motivating passive parties to participate in VFL can be challenging. In this paper, we focus on the problem of allocating incentives to the passive parties by the active party based on their contributions to the VFL process. We formulate this problem as a variant of the Nucleolus game theory concept, known as the Bankruptcy Problem, and solve it using the Talmud's division rule. We evaluate our proposed method on synthetic and real-world datasets and show that it ensures fairness and stability in incentive a
    
[^3]: 在协同学习和优化中激励竞争对手诚实行为的研究

    Incentivizing Honesty among Competitors in Collaborative Learning and Optimization. (arXiv:2305.16272v1 [cs.LG])

    [http://arxiv.org/abs/2305.16272](http://arxiv.org/abs/2305.16272)

    这项研究提出了一个模型来描述在协作学习中竞争对手的不诚实行为，提出了机制来激励诚实沟通，并确保学习质量与全面合作相当。

    

    协同学习技术能够让机器学习模型的训练比仅利用单一数据源的模型效果更好。然而，在许多情况下，潜在的参与者是下游任务中的竞争对手，如每个都希望通过提供最佳推荐来吸引客户的公司。这可能会激励不诚实的更新，损害其他参与者的模型，从而可能破坏协作的好处。在这项工作中，我们制定了一个模型来描述这种交互，并在该框架内研究了两个学习任务：单轮均值估计和强凸目标的多轮 SGD。对于一类自然的参与者行为，我们发现理性的客户会被激励强烈地操纵他们的更新，从而防止学习。然后，我们提出了机制来激励诚实沟通，并确保学习质量与全面合作相当。最后，我们通过实验证明了这一点。

    Collaborative learning techniques have the potential to enable training machine learning models that are superior to models trained on a single entity's data. However, in many cases, potential participants in such collaborative schemes are competitors on a downstream task, such as firms that each aim to attract customers by providing the best recommendations. This can incentivize dishonest updates that damage other participants' models, potentially undermining the benefits of collaboration. In this work, we formulate a game that models such interactions and study two learning tasks within this framework: single-round mean estimation and multi-round SGD on strongly-convex objectives. For a natural class of player actions, we show that rational clients are incentivized to strongly manipulate their updates, preventing learning. We then propose mechanisms that incentivize honest communication and ensure learning quality comparable to full cooperation. Lastly, we empirically demonstrate the
    

