# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Collaborative Inference via Dynamic Composition of Tiny AI Accelerators on MCUs.](http://arxiv.org/abs/2401.08637) | 该论文介绍了Synergy，一个通过动态组合微型AI加速器来进行协作推断的系统，有效地解决了在设备上AI需求不断增长时tinyML面临的关键挑战。Synergy通过提供虚拟计算空间和运行时编排模块，实现了资源的统一虚拟化视图和跨动态/异构加速器的最佳推断，其吞吐量平均提升了8.0倍。 |
| [^2] | [Collaborative Multi-Agent Heterogeneous Multi-Armed Bandits.](http://arxiv.org/abs/2305.18784) | 本研究研究了一个新的合作多智能体老虎机设置，并发展了去中心化算法以减少代理之间的集体遗憾，在数学分析中证明了该算法实现了近乎最优性能。 |

# 详细

[^1]: 通过MCU上微型AI加速器的动态组合实现协作推断

    Collaborative Inference via Dynamic Composition of Tiny AI Accelerators on MCUs. (arXiv:2401.08637v1 [cs.DC])

    [http://arxiv.org/abs/2401.08637](http://arxiv.org/abs/2401.08637)

    该论文介绍了Synergy，一个通过动态组合微型AI加速器来进行协作推断的系统，有效地解决了在设备上AI需求不断增长时tinyML面临的关键挑战。Synergy通过提供虚拟计算空间和运行时编排模块，实现了资源的统一虚拟化视图和跨动态/异构加速器的最佳推断，其吞吐量平均提升了8.0倍。

    

    微型AI加速器的出现为深度神经网络在极限边缘上的部署提供了机会，提供了较低的延迟、较低的功耗成本和改进的隐私保护。尽管取得了这些进展，但由于这些加速器的固有限制，如有限的内存和单设备焦点，仍存在挑战。本文介绍了Synergy，一个能够为多租户模型动态组合微型AI加速器的系统，有效解决了对于设备上AI的需求不断增长时tinyML面临的关键挑战。Synergy的一个关键特性是其提供了虚拟计算空间，为资源提供了统一的虚拟化视图，从而实现了对物理设备的高效任务映射。Synergy的运行时编排模块确保了跨动态和异构加速器的最佳推断。我们的评估结果显示，与基准相比，Synergy的吞吐量平均提升了8.0倍。

    The advent of tiny AI accelerators opens opportunities for deep neural network deployment at the extreme edge, offering reduced latency, lower power cost, and improved privacy in on-device ML inference. Despite these advancements, challenges persist due to inherent limitations of these accelerators, such as restricted onboard memory and single-device focus. This paper introduces Synergy, a system that dynamically composes tiny AI accelerators for multi-tenant models, effectively addressing tinyML's critical challenges for the increasing demand for on-device AI. A key feature of Synergy is its virtual computing space, providing a unified, virtualized view of resources and enabling efficient task mapping to physical devices. Synergy's runtime orchestration module ensures optimal inference across dynamic and heterogeneous accelerators. Our evaluations with 7 baselines and 8 models demonstrate that Synergy improves throughput by an average of 8.0X compared to baselines.
    
[^2]: 合作多智能体异构多臂老虎机翻译论文

    Collaborative Multi-Agent Heterogeneous Multi-Armed Bandits. (arXiv:2305.18784v1 [cs.LG])

    [http://arxiv.org/abs/2305.18784](http://arxiv.org/abs/2305.18784)

    本研究研究了一个新的合作多智能体老虎机设置，并发展了去中心化算法以减少代理之间的集体遗憾，在数学分析中证明了该算法实现了近乎最优性能。

    

    最近合作多智能体老虎机的研究吸引了很多关注。因此，我们开始研究一个新的合作设置，其中$N$个智能体中的每个智能体正在学习$M$个具有随机性的多臂老虎机，以减少他们的集体累计遗憾。我们开发了去中心化算法，促进了代理之间的合作，并针对两种情况进行了性能表征。通过推导每个代理的累积遗憾和集体遗憾的上限，我们对这些算法的性能进行了表征。我们还证明了这种情况下集体遗憾的下限，证明了所提出算法的近乎最优性能。

    The study of collaborative multi-agent bandits has attracted significant attention recently. In light of this, we initiate the study of a new collaborative setting, consisting of $N$ agents such that each agent is learning one of $M$ stochastic multi-armed bandits to minimize their group cumulative regret. We develop decentralized algorithms which facilitate collaboration between the agents under two scenarios. We characterize the performance of these algorithms by deriving the per agent cumulative regret and group regret upper bounds. We also prove lower bounds for the group regret in this setting, which demonstrates the near-optimal behavior of the proposed algorithms.
    

