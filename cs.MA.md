# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CMP: Cooperative Motion Prediction with Multi-Agent Communication](https://arxiv.org/abs/2403.17916) | 该论文提出了一种名为CMP的方法，利用LiDAR信号作为输入，通过合作感知和运动预测模块共享信息，解决了合作运动预测的问题。 |
| [^2] | [Networked Communication for Decentralised Agents in Mean-Field Games.](http://arxiv.org/abs/2306.02766) | 本研究在均场博弈中引入网络通信，提出了一种提高分布式智能体学习效率的方案，并进行了实际实验验证。 |

# 详细

[^1]: CMP：具有多智能体通信的合作运动预测

    CMP: Cooperative Motion Prediction with Multi-Agent Communication

    [https://arxiv.org/abs/2403.17916](https://arxiv.org/abs/2403.17916)

    该论文提出了一种名为CMP的方法，利用LiDAR信号作为输入，通过合作感知和运动预测模块共享信息，解决了合作运动预测的问题。

    

    随着自动驾驶车辆（AVs）的发展和车联网（V2X）通信的成熟，合作连接的自动化车辆（CAVs）的功能变得可能。本文基于合作感知，探讨了合作运动预测的可行性和有效性。我们的方法CMP以LiDAR信号作为输入，以增强跟踪和预测能力。与过去专注于合作感知或运动预测的工作不同，我们的框架是我们所知的第一个解决CAVs在感知和预测模块中共享信息的统一问题。我们的设计中还融入了能够容忍现实V2X带宽限制和传输延迟的独特能力，同时处理庞大的感知表示。我们还提出了预测聚合模块，统一了预测

    arXiv:2403.17916v1 Announce Type: cross  Abstract: The confluence of the advancement of Autonomous Vehicles (AVs) and the maturity of Vehicle-to-Everything (V2X) communication has enabled the capability of cooperative connected and automated vehicles (CAVs). Building on top of cooperative perception, this paper explores the feasibility and effectiveness of cooperative motion prediction. Our method, CMP, takes LiDAR signals as input to enhance tracking and prediction capabilities. Unlike previous work that focuses separately on either cooperative perception or motion prediction, our framework, to the best of our knowledge, is the first to address the unified problem where CAVs share information in both perception and prediction modules. Incorporated into our design is the unique capability to tolerate realistic V2X bandwidth limitations and transmission delays, while dealing with bulky perception representations. We also propose a prediction aggregation module, which unifies the predict
    
[^2]: 分布式智能体在均场博弈中的网络通信

    Networked Communication for Decentralised Agents in Mean-Field Games. (arXiv:2306.02766v2 [cs.MA] UPDATED)

    [http://arxiv.org/abs/2306.02766](http://arxiv.org/abs/2306.02766)

    本研究在均场博弈中引入网络通信，提出了一种提高分布式智能体学习效率的方案，并进行了实际实验验证。

    

    我们将网络通信引入均场博弈框架，特别是在无oracle的情况下，N个分布式智能体沿着经过的经验系统的单一非周期演化路径学习。我们证明，我们的架构在只有一些关于网络结构的合理假设的情况下，具有样本保证，在集中学习和独立学习情况之间有界。我们讨论了三个理论算法的样本保证实际上并不会导致实际收敛。因此，我们展示了在实际设置中，当理论参数未被观察到（导致Q函数的估计不准确）时，我们的通信方案显著加速了收敛速度，而无需依赖于一个不可取的集中式控制器的假设。我们对三个理论算法进行了几种实际的改进，使我们能够展示它们的第一个实证表现。

    We introduce networked communication to the mean-field game framework, in particular to oracle-free settings where $N$ decentralised agents learn along a single, non-episodic evolution path of the empirical system. We prove that our architecture, with only a few reasonable assumptions about network structure, has sample guarantees bounded between those of the centralised- and independent-learning cases. We discuss how the sample guarantees of the three theoretical algorithms do not actually result in practical convergence. Accordingly, we show that in practical settings where the theoretical parameters are not observed (leading to poor estimation of the Q-function), our communication scheme significantly accelerates convergence over the independent case, without relying on the undesirable assumption of a centralised controller. We contribute several further practical enhancements to all three theoretical algorithms, allowing us to showcase their first empirical demonstrations. Our expe
    

