# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FedSN: A General Federated Learning Framework over LEO Satellite Networks.](http://arxiv.org/abs/2311.01483) | FedSN是一个通用的联邦学习框架，用于解决在LEO卫星网络中的异构计算和存储能力、有限的上行速率以及模型陈旧等关键挑战。 |
| [^2] | [The Power of Populations in Decentralized Learning Dynamics.](http://arxiv.org/abs/2306.08670) | 本文研究了分散式学习动力学中个体群体的力量。我们介绍了一种分散式的多臂赌博机设置，并分析了几个针对此任务的分散式动力学家族。我们展示了这些动力学与一类“零和”乘法权重更新算法的联系，并开发了一个通用框架来分析这些协议的群体级遗憾。在广泛的参数范围下，我们得到了次线性的遗憾界限。 |
| [^3] | [FLEdge: Benchmarking Federated Machine Learning Applications in Edge Computing Systems.](http://arxiv.org/abs/2306.05172) | FLEdge是一个面向边缘计算系统中FL工作量的基准测试，通过研究硬件异构性、能量效率和隐私级别对FL系统训练的影响，以及客户端退出对最新FL策略的影响，提供了训练最先进的FL工作负载的新见解。 |

# 详细

[^1]: FedSN：一个适用于LEO卫星网络的通用联邦学习框架

    FedSN: A General Federated Learning Framework over LEO Satellite Networks. (arXiv:2311.01483v1 [cs.LG])

    [http://arxiv.org/abs/2311.01483](http://arxiv.org/abs/2311.01483)

    FedSN是一个通用的联邦学习框架，用于解决在LEO卫星网络中的异构计算和存储能力、有限的上行速率以及模型陈旧等关键挑战。

    

    最近，许多低地球轨道（LEO）卫星已经由商业公司成功地发射和部署到太空中，如SpaceX。由于LEO卫星配备了多模传感器，它们不仅用于通信，还用于各种机器学习应用，如空间调制识别、遥感图像分类等。然而，由于与LEO卫星的有限接触时间（例如5分钟），地面站（GS）可能无法下载如此大量的原始感测数据进行集中模型训练。因此，联邦学习（FL）已经成为解决这个问题的有希望的解决方案，通过在设备上进行训练。不幸的是，要在LEO卫星上使用FL，我们仍然面临三个关键挑战，即i）异构计算和存储能力，ii）有限的上行速率，以及iii）模型陈旧问题。为此，我们提出了一种名为FedSN的通用FL框架来解决上述挑战，一

    Recently, a large number of Low Earth Orbit (LEO) satellites have been launched and deployed successfully in space by commercial companies, such as SpaceX. Due to multimodal sensors equipped by the LEO satellites, they serve not only for communication but also for various machine learning applications, such as space modulation recognition, remote sensing image classification, etc. However, the ground station (GS) may be incapable of downloading such a large volume of raw sensing data for centralized model training due to the limited contact time with LEO satellites (e.g. 5 minutes). Therefore, federated learning (FL) has emerged as the promising solution to address this problem via on-device training. Unfortunately, to enable FL on LEO satellites, we still face three critical challenges that are i) heterogeneous computing and memory capabilities, ii) limited uplink rate, and iii) model staleness. To this end, we propose FedSN as a general FL framework to tackle the above challenges, an
    
[^2]: 分散式学习动力学中个体群体的力量

    The Power of Populations in Decentralized Learning Dynamics. (arXiv:2306.08670v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.08670](http://arxiv.org/abs/2306.08670)

    本文研究了分散式学习动力学中个体群体的力量。我们介绍了一种分散式的多臂赌博机设置，并分析了几个针对此任务的分散式动力学家族。我们展示了这些动力学与一类“零和”乘法权重更新算法的联系，并开发了一个通用框架来分析这些协议的群体级遗憾。在广泛的参数范围下，我们得到了次线性的遗憾界限。

    

    我们研究了一种分散式多臂赌博机设置，在一个由$n$个受内存限制的节点组成的种群中，采用了谣言模型：每轮，每个节点本地采用$m$个臂之一，观察从臂的（对抗选择的）分布中抽取的奖励，然后与随机抽取的邻居进行通信，交换信息，以确定下一轮的策略。我们介绍并分析了几个针对此任务的分散式动力学家族：每个节点的决策完全是局部的，只依赖于其最新获得的奖励以及它抽样的邻居的奖励。我们展示了这些分散式动力学的全局演化与特定类型的“零和”乘法权重更新算法之间的联系，并且开发了一个分析这些自然协议的群体级遗憾的通用框架。利用这个框架，我们在广泛的参数范围（即，种群的大小和nu的大小）下推导了次线性遗憾界限。

    We study a distributed multi-armed bandit setting among a population of $n$ memory-constrained nodes in the gossip model: at each round, every node locally adopts one of $m$ arms, observes a reward drawn from the arm's (adversarially chosen) distribution, and then communicates with a randomly sampled neighbor, exchanging information to determine its policy in the next round. We introduce and analyze several families of dynamics for this task that are decentralized: each node's decision is entirely local and depends only on its most recently obtained reward and that of the neighbor it sampled. We show a connection between the global evolution of these decentralized dynamics with a certain class of "zero-sum" multiplicative weights update algorithms, and we develop a general framework for analyzing the population-level regret of these natural protocols. Using this framework, we derive sublinear regret bounds under a wide range of parameter regimes (i.e., the size of the population and nu
    
[^3]: FLEdge：边缘计算系统中联邦机器学习应用的基准测试

    FLEdge: Benchmarking Federated Machine Learning Applications in Edge Computing Systems. (arXiv:2306.05172v1 [cs.LG])

    [http://arxiv.org/abs/2306.05172](http://arxiv.org/abs/2306.05172)

    FLEdge是一个面向边缘计算系统中FL工作量的基准测试，通过研究硬件异构性、能量效率和隐私级别对FL系统训练的影响，以及客户端退出对最新FL策略的影响，提供了训练最先进的FL工作负载的新见解。

    

    近年来，联邦机器学习（FL）备受关注。 FL基准测试主要在模拟系统或数据中心环境中进行探索，忽略了与边缘计算密切相关的实际系统设置。 我们通过引入面向边缘计算系统中FL工作量的基准测试FLEdge来弥补这一研究差距。我们系统地研究了硬件异构性、训练过程中的能量效率以及各种不同隐私级别对FL系统训练的影响。为了使这个基准测试适用于实际场景，我们评估了客户端退出对具有高达50％失效率的最新FL策略的影响。 FLEdge提供了新的见解，例如，在旧GPU加速的嵌入式设备上训练最先进的FL工作负载比在现代服务器级GPU上训练高达3倍的能量效率。

    Federated Machine Learning (FL) has received considerable attention in recent years. FL benchmarks are predominantly explored in either simulated systems or data center environments, neglecting the setups of real-world systems, which are often closely linked to edge computing. We close this research gap by introducing FLEdge, a benchmark targeting FL workloads in edge computing systems. We systematically study hardware heterogeneity, energy efficiency during training, and the effect of various differential privacy levels on training in FL systems. To make this benchmark applicable to real-world scenarios, we evaluate the impact of client dropouts on state-of-the-art FL strategies with failure rates as high as 50%. FLEdge provides new insights, such as that training state-of-the-art FL workloads on older GPU-accelerated embedded devices is up to 3x more energy efficient than on modern server-grade GPUs.
    

