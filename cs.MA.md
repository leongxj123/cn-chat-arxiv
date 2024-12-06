# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Facility Location Games with Scaling Effects](https://arxiv.org/abs/2402.18908) | 研究了具有规模效应的设施选址游戏，提供了对于连续比例函数和分段线性比例函数的结果，适用于许多实际情景，同时探讨了近似机制设计设置下代理可能不再单峰偏好的条件与成本近似比率。 |
| [^2] | [Network Formation and Dynamics Among Multi-LLMs](https://arxiv.org/abs/2402.10659) | 分析了多个LLM在社交网络中的行为，发现它们在给定网络结构并被询问形成网络偏好时表现出与人类社交动态一致的原则。 |
| [^3] | [A Deep RL Approach on Task Placement and Scaling of Edge Resources for Cellular Vehicle-to-Network Service Provisioning.](http://arxiv.org/abs/2305.09832) | 本文提出了一种基于深度强化学习的分散式方法，用于解决车联网服务提供中的任务部署和边缘资源的扩展问题。 |

# 详细

[^1]: 具有规模效应的设施选址游戏

    Facility Location Games with Scaling Effects

    [https://arxiv.org/abs/2402.18908](https://arxiv.org/abs/2402.18908)

    研究了具有规模效应的设施选址游戏，提供了对于连续比例函数和分段线性比例函数的结果，适用于许多实际情景，同时探讨了近似机制设计设置下代理可能不再单峰偏好的条件与成本近似比率。

    

    我们考虑了经典的设施选址问题的一个变种，其中每个代理的个人成本函数等于他们距离设施的距离乘以一个由设施位置确定的比例因子。除了一般类别的连续比例函数外，我们还提供了适用于许多实际情景的比例函数的分段线性比例函数的结果。我们关注总成本和最大成本的目标，并描述了最优解的计算。然后我们转向近似机制设计设置，观察到代理的偏好可能不再是单峰的。因此，我们表征了确保代理具有单峰偏好的比例函数条件。在这些条件下，我们找到了能够通过strategyproof和anonymous me达到的总成本和最大成本近似比率的结果。

    arXiv:2402.18908v1 Announce Type: cross  Abstract: We take the classic facility location problem and consider a variation, in which each agent's individual cost function is equal to their distance from the facility multiplied by a scaling factor which is determined by the facility placement. In addition to the general class of continuous scaling functions, we also provide results for piecewise linear scaling functions which can effectively approximate or model the scaling of many real world scenarios. We focus on the objectives of total and maximum cost, describing the computation of the optimal solution. We then move to the approximate mechanism design setting, observing that the agents' preferences may no longer be single-peaked. Consequently, we characterize the conditions on scaling functions which ensure that agents have single-peaked preferences. Under these conditions, we find results on the total and maximum cost approximation ratios achievable by strategyproof and anonymous me
    
[^2]: 多个LLM之间的网络形成与动态

    Network Formation and Dynamics Among Multi-LLMs

    [https://arxiv.org/abs/2402.10659](https://arxiv.org/abs/2402.10659)

    分析了多个LLM在社交网络中的行为，发现它们在给定网络结构并被询问形成网络偏好时表现出与人类社交动态一致的原则。

    

    社交网络影响行为、偏好和关系，在人类社会中对信息和规范的传播起着至关重要的作用。随着大型语言模型（LLMs）越来越多地融入社交和专业环境中，理解它们在社交网络和互动背景下的行为变得至关重要。我们的研究分析了标准网络结构和现实世界网络的行为，以确定多个LLMs的动态是否与人类社交动态一致。我们探讨了各种社交网络原则，包括微观层面的概念，如偏爱附着、三角闭合和同似性，以及宏观层面的概念，如社区结构和小世界现象。我们的研究发现表明，当向LLMs提供网络结构并询问它们对网络形成的偏好时，它们表现出所有这些原则。

    arXiv:2402.10659v1 Announce Type: cross  Abstract: Social networks influence behaviors, preferences, and relationships and play a crucial role in the dissemination of information and norms within human societies. As large language models (LLMs) increasingly integrate into social and professional environments, understanding their behavior within the context of social networks and interactions becomes essential. Our study analyzes the behaviors of standard network structures and real-world networks to determine whether the dynamics of multiple LLMs align with human social dynamics. We explore various social network principles, including micro-level concepts such as preferential attachment, triadic closure, and homophily, as well as macro-level concepts like community structure and the small-world phenomenon. Our findings suggest that LLMs demonstrate all these principles when they are provided with network structures and asked about their preferences regarding network formation. Furtherm
    
[^3]: 基于深度强化学习的边缘资源任务部署和扩展方法用于车载网络服务提供

    A Deep RL Approach on Task Placement and Scaling of Edge Resources for Cellular Vehicle-to-Network Service Provisioning. (arXiv:2305.09832v1 [cs.AI])

    [http://arxiv.org/abs/2305.09832](http://arxiv.org/abs/2305.09832)

    本文提出了一种基于深度强化学习的分散式方法，用于解决车联网服务提供中的任务部署和边缘资源的扩展问题。

    

    “车联网”正处于我们社会数字化转型的前沿。本文提出了一种分散式方法用于提供车辆通联网（C-V2N）服务，解决服务任务部署和边缘资源的扩展问题。我们证明了这个联合问题的复杂性，并提出了一个两个问题的联接方式，采用了基于贪心算法的关于任务部署的方法和基于 Deep Deterministic Policy Gradient (DDPG) 的扩展方法。本文还对我们的方法进行了基准测试，重点关注了扩展代理与多个状态下最先进的扩展方法的性能比较。

    Cellular-Vehicle-to-Everything (C-V2X) is currently at the forefront of the digital transformation of our society. By enabling vehicles to communicate with each other and with the traffic environment using cellular networks, we redefine transportation, improving road safety and transportation services, increasing efficiency of traffic flows, and reducing environmental impact. This paper proposes a decentralized approach for provisioning Cellular Vehicular-to-Network (C-V2N) services, addressing the coupled problems of service task placement and scaling of edge resources. We formalize the joint problem and prove its complexity. We propose an approach to tackle it, linking the two problems, employing decentralized decision-making using (i) a greedy approach for task placement and (ii) a Deep Deterministic Policy Gradient (DDPG) based approach for scaling. We benchmark the performance of our approach, focusing on the scaling agent, against several State-of-the-Art (SoA) scaling approaches
    

