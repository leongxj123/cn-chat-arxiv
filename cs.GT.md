# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Discovering How Agents Learn Using Few Data.](http://arxiv.org/abs/2307.06640) | 本研究提出了一个在少量数据上识别智能体学习动力学的算法框架。通过使用多项式回归和引入捕捉智能体行为的侧信息约束，该方法可以在使用单个轨迹的短暂运行中仅仅5个样本的情况下准确恢复真实动力学。 |

# 详细

[^1]: 发现智能体在少量数据上的学习方式

    Discovering How Agents Learn Using Few Data. (arXiv:2307.06640v1 [cs.GT])

    [http://arxiv.org/abs/2307.06640](http://arxiv.org/abs/2307.06640)

    本研究提出了一个在少量数据上识别智能体学习动力学的算法框架。通过使用多项式回归和引入捕捉智能体行为的侧信息约束，该方法可以在使用单个轨迹的短暂运行中仅仅5个样本的情况下准确恢复真实动力学。

    

    分散式学习算法是设计多智能体系统的重要工具，它们使得智能体能够从经验和过去的交互中自主学习。本研究提出了一个理论和算法框架，用于实时识别规定智能体行为的学习动力学，只需使用单个系统轨迹的短暂突发。我们的方法通过多项式回归识别智能体动力学，在有限的数据上通过引入捕捉智能体行为的基本假设或期望的侧信息约束来补偿，并且通过使用和优化约束的平方和计算来实施这些约束，从而得到越来越准确的智能体动力学近似层次。广泛的实验证明，我们的方法只使用单个轨迹的短暂运行中的5个样本，就可以准确恢复各种基准测试中的真实动力学，包括均衡选择和预测。

    Decentralized learning algorithms are an essential tool for designing multi-agent systems, as they enable agents to autonomously learn from their experience and past interactions. In this work, we propose a theoretical and algorithmic framework for real-time identification of the learning dynamics that govern agent behavior using a short burst of a single system trajectory. Our method identifies agent dynamics through polynomial regression, where we compensate for limited data by incorporating side-information constraints that capture fundamental assumptions or expectations about agent behavior. These constraints are enforced computationally using sum-of-squares optimization, leading to a hierarchy of increasingly better approximations of the true agent dynamics. Extensive experiments demonstrated that our approach, using only 5 samples from a short run of a single trajectory, accurately recovers the true dynamics across various benchmarks, including equilibrium selection and predictio
    

