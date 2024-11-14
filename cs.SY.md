# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Morphological Symmetries in Robotics](https://arxiv.org/abs/2402.15552) | 形态对称性是机器人系统中的固有性质，通过对运动结构和质量的对称分布，延伸至机器人状态空间和传感器测量，进而影响机器人的运动方程和最优控制策略，并在机器人学建模、控制和设计中具有重要意义。 |
| [^2] | [Controlling Large Electric Vehicle Charging Stations via User Behavior Modeling and Stochastic Programming](https://arxiv.org/abs/2402.13224) | 本文介绍了一个新的电动汽车充电站模型，通过用户行为建模和随机规划，解决了充电会话不确定性问题，并提出了两种方法来优化成本并提高用户满意度。 |
| [^3] | [Reinforcement Learning (RL) Augmented Cold Start Frequency Reduction in Serverless Computing.](http://arxiv.org/abs/2308.07541) | 本文提出了一种基于强化学习的方法来降低无服务器计算中的冷启动频率。通过使用Q学习和考虑多种指标，我们可以在预期需求的基础上提前初始化函数，从而减少冷启动次数。 |

# 详细

[^1]: 机器人学中的形态对称性

    Morphological Symmetries in Robotics

    [https://arxiv.org/abs/2402.15552](https://arxiv.org/abs/2402.15552)

    形态对称性是机器人系统中的固有性质，通过对运动结构和质量的对称分布，延伸至机器人状态空间和传感器测量，进而影响机器人的运动方程和最优控制策略，并在机器人学建模、控制和设计中具有重要意义。

    

    我们提出了一个全面的框架来研究和利用机器人系统中的形态对称性。这些是机器人形态的固有特性，经常在动物生物学和机器人学中观察到，源于运动结构的复制和质量的对称分布。我们说明了这些对称性如何延伸到机器人的状态空间以及本体感知和外部感知传感器测量，导致机器人的运动方程和最优控制策略的等不变性。因此，我们认识到形态对称性作为一个相关且以前未被探索的受物理启示的几何先验，对机器人建模、控制、估计和设计中使用的数据驱动和分析方法都具有重要影响。对于数据驱动方法，我们演示了形态对称性如何提高机器学习模型的样本效率和泛化能力

    arXiv:2402.15552v1 Announce Type: cross  Abstract: We present a comprehensive framework for studying and leveraging morphological symmetries in robotic systems. These are intrinsic properties of the robot's morphology, frequently observed in animal biology and robotics, which stem from the replication of kinematic structures and the symmetrical distribution of mass. We illustrate how these symmetries extend to the robot's state space and both proprioceptive and exteroceptive sensor measurements, resulting in the equivariance of the robot's equations of motion and optimal control policies. Thus, we recognize morphological symmetries as a relevant and previously unexplored physics-informed geometric prior, with significant implications for both data-driven and analytical methods used in modeling, control, estimation and design in robotics. For data-driven methods, we demonstrate that morphological symmetries can enhance the sample efficiency and generalization of machine learning models 
    
[^2]: 通过用户行为建模和随机规划控制大型电动汽车充电站

    Controlling Large Electric Vehicle Charging Stations via User Behavior Modeling and Stochastic Programming

    [https://arxiv.org/abs/2402.13224](https://arxiv.org/abs/2402.13224)

    本文介绍了一个新的电动汽车充电站模型，通过用户行为建模和随机规划，解决了充电会话不确定性问题，并提出了两种方法来优化成本并提高用户满意度。

    

    本文介绍了一个电动汽车充电站（EVCS）模型，该模型融合了真实世界的约束条件，如插槽功率限制、合同阈值超限惩罚以及电动汽车（EVs）的早期断开。我们提出了一个在不确定性下控制EVCS的问题形式，并实施了两种多阶段随机规划方法，利用用户提供的信息，即模型预测控制和二阶段随机规划。该模型解决了充电会话开始和结束时间以及能量需求的不确定性。基于驻留时间依赖随机过程的用户行为模型增强了成本降低的同时保持客户满意度。通过使用真实世界数据集进行的22天模拟展示了两种提出方法相对于两个基线的优势。两阶段方法证明了针对早期断开的鲁棒性，考虑了更多

    arXiv:2402.13224v1 Announce Type: cross  Abstract: This paper introduces an Electric Vehicle Charging Station (EVCS) model that incorporates real-world constraints, such as slot power limitations, contract threshold overruns penalties, or early disconnections of electric vehicles (EVs). We propose a formulation of the problem of EVCS control under uncertainty, and implement two Multi-Stage Stochastic Programming approaches that leverage user-provided information, namely, Model Predictive Control and Two-Stage Stochastic Programming. The model addresses uncertainties in charging session start and end times, as well as in energy demand. A user's behavior model based on a sojourn-time-dependent stochastic process enhances cost reduction while maintaining customer satisfaction. The benefits of the two proposed methods are showcased against two baselines over a 22-day simulation using a real-world dataset. The two-stage approach proves robust against early disconnections, considering a more
    
[^3]: 基于强化学习的无服务器计算中冷启动频率降低方法

    Reinforcement Learning (RL) Augmented Cold Start Frequency Reduction in Serverless Computing. (arXiv:2308.07541v1 [cs.DC])

    [http://arxiv.org/abs/2308.07541](http://arxiv.org/abs/2308.07541)

    本文提出了一种基于强化学习的方法来降低无服务器计算中的冷启动频率。通过使用Q学习和考虑多种指标，我们可以在预期需求的基础上提前初始化函数，从而减少冷启动次数。

    

    函数即服务是一种云计算范例，为应用程序提供了事件驱动执行模型。它通过从开发者那里消除资源管理责任，提供透明和按需可扩展性来实现无服务器特性。典型的无服务器应用程序对响应时间和可扩展性有严格要求，因此依赖于部署的服务为客户提供快速和容错的反馈。然而，函数即服务范例在需要按需初始化函数时存在非常可观的延迟，即冷启动问题。本研究旨在通过使用强化学习来减少平台上的冷启动频率。我们的方法使用Q学习，并考虑函数的CPU利用率、已有函数实例和响应失败率等指标，根据预期需求提前主动初始化函数。我们提出的解决方案在Kubeless上实现并进行评估。

    Function-as-a-Service is a cloud computing paradigm offering an event-driven execution model to applications. It features serverless attributes by eliminating resource management responsibilities from developers and offers transparent and on-demand scalability of applications. Typical serverless applications have stringent response time and scalability requirements and therefore rely on deployed services to provide quick and fault-tolerant feedback to clients. However, the FaaS paradigm suffers from cold starts as there is a non-negligible delay associated with on-demand function initialization. This work focuses on reducing the frequency of cold starts on the platform by using Reinforcement Learning. Our approach uses Q-learning and considers metrics such as function CPU utilization, existing function instances, and response failure rate to proactively initialize functions in advance based on the expected demand. The proposed solution was implemented on Kubeless and was evaluated usin
    

