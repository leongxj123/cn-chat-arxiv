# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sample-Efficient Co-Design of Robotic Agents Using Multi-fidelity Training on Universal Policy Network.](http://arxiv.org/abs/2309.04085) | 该论文提出了一种使用多保真度训练的机器人代理共设计的样本高效方法，通过在设计空间中共享学习到的控制器，以及通过特定方式遍历设计矩阵，可以提高设计评估的效率。 |
| [^2] | [Learning to Control and Coordinate Mixed Traffic Through Robot Vehicles at Complex and Unsignalized Intersections.](http://arxiv.org/abs/2301.05294) | 本研究提出了一种去中心化的多智能体强化学习方法，用于控制和协调混合交通，特别是人驾驶车辆和机器人车辆在实际复杂交叉口的应用。实验结果表明，使用5%的机器人车辆可以有效防止交叉口内的拥堵形成。 |

# 详细

[^1]: 使用多保真度训练在通用策略网络上的机器人代理共设计的样本高效方法

    Sample-Efficient Co-Design of Robotic Agents Using Multi-fidelity Training on Universal Policy Network. (arXiv:2309.04085v1 [cs.RO])

    [http://arxiv.org/abs/2309.04085](http://arxiv.org/abs/2309.04085)

    该论文提出了一种使用多保真度训练的机器人代理共设计的样本高效方法，通过在设计空间中共享学习到的控制器，以及通过特定方式遍历设计矩阵，可以提高设计评估的效率。

    

    共设计涉及同时优化控制器和代理物理设计。其固有的双层优化形式要求通过内层控制优化来驱动外层设计优化。当设计空间较大且每个设计评估都涉及数据密集型的强化学习过程时，这可能会带来挑战。为了提高样本效率，我们提出了一种基于Hyperband的多保真度设计探索策略，在设计空间中通过通用策略学习者将学习到的控制器进行关联，以启动后续控制器学习问题。此外，我们推荐一种特定的遍历Hyperband生成的设计矩阵的方式，以确保随着每个新的设计评估，通用策略学习者的增强效果越来越强，从而降低Hyperband的随机性。实验证明了该方法在广泛的年龄范围内的表现。

    Co-design involves simultaneously optimizing the controller and agents physical design. Its inherent bi-level optimization formulation necessitates an outer loop design optimization driven by an inner loop control optimization. This can be challenging when the design space is large and each design evaluation involves data-intensive reinforcement learning process for control optimization. To improve the sample-efficiency we propose a multi-fidelity-based design exploration strategy based on Hyperband where we tie the controllers learnt across the design spaces through a universal policy learner for warm-starting the subsequent controller learning problems. Further, we recommend a particular way of traversing the Hyperband generated design matrix that ensures that the stochasticity of the Hyperband is reduced the most with the increasing warm starting effect of the universal policy learner as it is strengthened with each new design evaluation. Experiments performed on a wide range of age
    
[^2]: 通过机器人车辆在复杂和无信号的交叉口中学习控制和协调混合交通

    Learning to Control and Coordinate Mixed Traffic Through Robot Vehicles at Complex and Unsignalized Intersections. (arXiv:2301.05294v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.05294](http://arxiv.org/abs/2301.05294)

    本研究提出了一种去中心化的多智能体强化学习方法，用于控制和协调混合交通，特别是人驾驶车辆和机器人车辆在实际复杂交叉口的应用。实验结果表明，使用5%的机器人车辆可以有效防止交叉口内的拥堵形成。

    

    交叉口是现代大都市交通中必不可少的道路基础设施。然而，由于交通事故或缺乏交通协调机制（如交通信号灯），它们也可能成为交通流的瓶颈。最近，提出了各种超越传统控制方法的控制和协调机制，以提高交叉口交通的效率。在这些方法中，控制可预见的包含人驾驶车辆（HVs）和机器人车辆（RVs）的混合交通已经出现。在本项目中，我们提出了一种去中心化的多智能体强化学习方法，用于实际复杂交叉口的混合交通的控制和协调，这是一个以前未被探索过的主题。我们进行了全面的实验，以展示我们方法的有效性。特别是，我们展示了在实际交通条件下，使用5%的RVs，我们可以防止复杂交叉口内的拥堵形成。

    Intersections are essential road infrastructures for traffic in modern metropolises. However, they can also be the bottleneck of traffic flows as a result of traffic incidents or the absence of traffic coordination mechanisms such as traffic lights. Recently, various control and coordination mechanisms that are beyond traditional control methods have been proposed to improve the efficiency of intersection traffic. Amongst these methods, the control of foreseeable mixed traffic that consists of human-driven vehicles (HVs) and robot vehicles (RVs) has emerged. In this project, we propose a decentralized multi-agent reinforcement learning approach for the control and coordination of mixed traffic at real-world, complex intersections--a topic that has not been previously explored. Comprehensive experiments are conducted to show the effectiveness of our approach. In particular, we show that using 5% RVs, we can prevent congestion formation inside a complex intersection under the actual traf
    

