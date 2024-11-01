# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Temporally Layered Architecture for Efficient Continuous Control.](http://arxiv.org/abs/2305.18701) | 这项研究提出了一种时间分层架构，利用时间抽象实现了高效的连续控制，具有节能、持续探索、减少决策、减少抖动和增加动作重复等优势。 |
| [^2] | [Knowledge Equivalence in Digital Twins of Intelligent Systems.](http://arxiv.org/abs/2204.07481) | 本文研究了智能系统数字孪生中的知识等价性，提出了在模型与物理系统之间同步知识的新技术。 |

# 详细

[^1]: 高效连续控制的时间分层架构

    Temporally Layered Architecture for Efficient Continuous Control. (arXiv:2305.18701v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2305.18701](http://arxiv.org/abs/2305.18701)

    这项研究提出了一种时间分层架构，利用时间抽象实现了高效的连续控制，具有节能、持续探索、减少决策、减少抖动和增加动作重复等优势。

    

    我们提出了一种时间分层架构（TLA），用于具有最小能量消耗的时间自适应控制。TLA将快速策略和慢速策略层叠在一起，实现时间抽象，使每个层可以专注于不同的时间尺度。我们的设计借鉴了人脑的节能机制，根据环境需求在不同的时间尺度上执行动作。我们证明了TLA除了节能之外，还提供了许多额外的优势，包括持续探索、减少需求决策、减少抖动和增加动作重复。我们在一系列连续控制任务上评估了我们的方法，并展示了TLA在多个重要指标上相对于现有方法的显著优势。我们还引入了多目标评分来定性评估连续控制策略，并展示了TLA具有显著更好的评分。我们的训练算法在慢速策略和

    We present a temporally layered architecture (TLA) for temporally adaptive control with minimal energy expenditure. The TLA layers a fast and a slow policy together to achieve temporal abstraction that allows each layer to focus on a different time scale. Our design draws on the energy-saving mechanism of the human brain, which executes actions at different timescales depending on the environment's demands. We demonstrate that beyond energy saving, TLA provides many additional advantages, including persistent exploration, fewer required decisions, reduced jerk, and increased action repetition. We evaluate our method on a suite of continuous control tasks and demonstrate the significant advantages of TLA over existing methods when measured over multiple important metrics. We also introduce a multi-objective score to qualitatively assess continuous control policies and demonstrate a significantly better score for TLA. Our training algorithm uses minimal communication between the slow and
    
[^2]: 智能系统数字孪生中的知识等价性

    Knowledge Equivalence in Digital Twins of Intelligent Systems. (arXiv:2204.07481v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2204.07481](http://arxiv.org/abs/2204.07481)

    本文研究了智能系统数字孪生中的知识等价性，提出了在模型与物理系统之间同步知识的新技术。

    

    数字孪生包含了对所研究的物理世界进行实时数据驱动建模的模型，并且可以使用模拟来优化物理世界。然而，数字孪生所做的分析只有在模型与实际物理世界等价的情况下才是有效和可靠的。在物理系统是智能和自主的情况下，保持这样一个等价模型具有挑战性。本文特别关注智能系统数字孪生模型，其中系统具有知识感知能力但能力有限。数字孪生通过在模拟环境中积累更多知识，从元层面上改进了物理系统的行为。在虚拟空间中复制这样一个智能物理系统的知识感知能力需要采用新颖的等价性维护技术，特别是在模型与物理系统之间同步知识。本文提出了知识等价性的概念。

    A digital twin contains up-to-date data-driven models of the physical world being studied and can use simulation to optimise the physical world. However, the analysis made by the digital twin is valid and reliable only when the model is equivalent to the physical world. Maintaining such an equivalent model is challenging, especially when the physical systems being modelled are intelligent and autonomous. The paper focuses in particular on digital twin models of intelligent systems where the systems are knowledge-aware but with limited capability. The digital twin improves the acting of the physical system at a meta-level by accumulating more knowledge in the simulated environment. The modelling of such an intelligent physical system requires replicating the knowledge-awareness capability in the virtual space. Novel equivalence maintaining techniques are needed, especially in synchronising the knowledge between the model and the physical system. This paper proposes the notion of knowled
    

