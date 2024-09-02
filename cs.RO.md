# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scalable Multi-Agent Reinforcement Learning for Warehouse Logistics with Robotic and Human Co-Workers.](http://arxiv.org/abs/2212.11498) | 该论文提出了一种可扩展的多智能体强化学习方法，用于仓库物流中的机器人和人类同事合作。他们通过分层的MARL算法，让经理和工人代理根据全局目标进行协同训练，以最大化拣货速率。 |

# 详细

[^1]: 可扩展的多智能体强化学习在仓库物流中与机器人和人类同事合作

    Scalable Multi-Agent Reinforcement Learning for Warehouse Logistics with Robotic and Human Co-Workers. (arXiv:2212.11498v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.11498](http://arxiv.org/abs/2212.11498)

    该论文提出了一种可扩展的多智能体强化学习方法，用于仓库物流中的机器人和人类同事合作。他们通过分层的MARL算法，让经理和工人代理根据全局目标进行协同训练，以最大化拣货速率。

    

    我们设想一个仓库里有数十个移动机器人和人类分拣员一起工作，收集和交付仓库内的物品。我们要解决的基本问题是称为拣货问题，即这些工作代理人如何在仓库中协调他们的移动和行为以最大化性能（例如订单吞吐量）。传统的行业方法使用启发式方法需要大量的工程努力来为固有可变的仓库配置进行优化。相比之下，多智能体强化学习（MARL）可以灵活地应用于不同的仓库配置（例如大小，布局，工人数量/类型，物品补充频率），因为代理人通过经验学习如何最优地相互合作。我们开发了分层MARL算法，其中一个管理者为工人代理分配目标，并且管理者和工人的策略被共同训练以最大化全局目标（例如拣货速率）。

    We envision a warehouse in which dozens of mobile robots and human pickers work together to collect and deliver items within the warehouse. The fundamental problem we tackle, called the order-picking problem, is how these worker agents must coordinate their movement and actions in the warehouse to maximise performance (e.g. order throughput). Established industry methods using heuristic approaches require large engineering efforts to optimise for innately variable warehouse configurations. In contrast, multi-agent reinforcement learning (MARL) can be flexibly applied to diverse warehouse configurations (e.g. size, layout, number/types of workers, item replenishment frequency), as the agents learn through experience how to optimally cooperate with one another. We develop hierarchical MARL algorithms in which a manager assigns goals to worker agents, and the policies of the manager and workers are co-trained toward maximising a global objective (e.g. pick rate). Our hierarchical algorith
    

