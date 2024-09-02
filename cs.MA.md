# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Emergence of Social Norms in Large Language Model-based Agent Societies](https://arxiv.org/abs/2403.08251) | 提出了第一个赋予大型语言模型Agent群体内社会规范出现的生成式Agent架构CRSEC，实验证明其能力。 |
| [^2] | [Optimizing Agent Collaboration through Heuristic Multi-Agent Planning.](http://arxiv.org/abs/2301.01246) | 提出了一种启发式多智能体规划算法，解决了涉及不同类型智能体的问题，比现有算法表现更好。 |
| [^3] | [Scalable Multi-Agent Reinforcement Learning for Warehouse Logistics with Robotic and Human Co-Workers.](http://arxiv.org/abs/2212.11498) | 该论文提出了一种可扩展的多智能体强化学习方法，用于仓库物流中的机器人和人类同事合作。他们通过分层的MARL算法，让经理和工人代理根据全局目标进行协同训练，以最大化拣货速率。 |

# 详细

[^1]: 基于大型语言模型的Agent社会中社会规范的出现

    Emergence of Social Norms in Large Language Model-based Agent Societies

    [https://arxiv.org/abs/2403.08251](https://arxiv.org/abs/2403.08251)

    提出了第一个赋予大型语言模型Agent群体内社会规范出现的生成式Agent架构CRSEC，实验证明其能力。

    

    社会规范的出现吸引了社会科学、认知科学以及人工智能等各个领域的广泛关注。本文提出了第一个赋予大型语言模型Agent群体内社会规范出现的生成式Agent架构CRSEC。我们的架构包括四个模块：Creation & Representation、Spreading、Evaluation和Compliance。我们的架构处理了几个关键方面的紧急过程：(i)社会规范的来源，(ii)它们如何被正式表示，(iii)它们如何通过Agent的交流和观察传播，(iv)如何通过合理检查进行检查并在长期内进行综合，(v)如何被纳入Agent的计划和行动中。我们在Smallville沙盒游戏环境中进行的实验展示了我们的架构的能力。

    arXiv:2403.08251v1 Announce Type: cross  Abstract: The emergence of social norms has attracted much interest in a wide array of disciplines, ranging from social science and cognitive science to artificial intelligence. In this paper, we propose the first generative agent architecture that empowers the emergence of social norms within a population of large language model-based agents. Our architecture, named CRSEC, consists of four modules: Creation & Representation, Spreading, Evaluation, and Compliance. Our architecture addresses several important aspects of the emergent processes all in one: (i) where social norms come from, (ii) how they are formally represented, (iii) how they spread through agents' communications and observations, (iv) how they are examined with a sanity check and synthesized in the long term, and (v) how they are incorporated into agents' planning and actions. Our experiments deployed in the Smallville sandbox game environment demonstrate the capability of our ar
    
[^2]: 启发式多智能体规划优化智能体协作

    Optimizing Agent Collaboration through Heuristic Multi-Agent Planning. (arXiv:2301.01246v3 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2301.01246](http://arxiv.org/abs/2301.01246)

    提出了一种启发式多智能体规划算法，解决了涉及不同类型智能体的问题，比现有算法表现更好。

    

    针对涉及到不同类型感知智能体的问题，目前解决QDec-POMDP的SOTA算法QDec-FP和QDec-FPS无法有效解决。本文提出了一种新算法，通过要求智能体采取相同的计划，以解决这个问题。在这些情况下，我们的算法比QDec-FP和QDec-FPS都表现更好。

    The SOTA algorithms for addressing QDec-POMDP issues, QDec-FP and QDec-FPS, are unable to effectively tackle problems that involve different types of sensing agents. We propose a new algorithm that addresses this issue by requiring agents to adopt the same plan if one agent is unable to take a sensing action but the other can. Our algorithm performs significantly better than both QDec-FP and QDec-FPS in these types of situations.
    
[^3]: 可扩展的多智能体强化学习在仓库物流中与机器人和人类同事合作

    Scalable Multi-Agent Reinforcement Learning for Warehouse Logistics with Robotic and Human Co-Workers. (arXiv:2212.11498v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.11498](http://arxiv.org/abs/2212.11498)

    该论文提出了一种可扩展的多智能体强化学习方法，用于仓库物流中的机器人和人类同事合作。他们通过分层的MARL算法，让经理和工人代理根据全局目标进行协同训练，以最大化拣货速率。

    

    我们设想一个仓库里有数十个移动机器人和人类分拣员一起工作，收集和交付仓库内的物品。我们要解决的基本问题是称为拣货问题，即这些工作代理人如何在仓库中协调他们的移动和行为以最大化性能（例如订单吞吐量）。传统的行业方法使用启发式方法需要大量的工程努力来为固有可变的仓库配置进行优化。相比之下，多智能体强化学习（MARL）可以灵活地应用于不同的仓库配置（例如大小，布局，工人数量/类型，物品补充频率），因为代理人通过经验学习如何最优地相互合作。我们开发了分层MARL算法，其中一个管理者为工人代理分配目标，并且管理者和工人的策略被共同训练以最大化全局目标（例如拣货速率）。

    We envision a warehouse in which dozens of mobile robots and human pickers work together to collect and deliver items within the warehouse. The fundamental problem we tackle, called the order-picking problem, is how these worker agents must coordinate their movement and actions in the warehouse to maximise performance (e.g. order throughput). Established industry methods using heuristic approaches require large engineering efforts to optimise for innately variable warehouse configurations. In contrast, multi-agent reinforcement learning (MARL) can be flexibly applied to diverse warehouse configurations (e.g. size, layout, number/types of workers, item replenishment frequency), as the agents learn through experience how to optimally cooperate with one another. We develop hierarchical MARL algorithms in which a manager assigns goals to worker agents, and the policies of the manager and workers are co-trained toward maximising a global objective (e.g. pick rate). Our hierarchical algorith
    

