# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Resource-Aware Hierarchical Federated Learning in Wireless Video Caching Networks](https://arxiv.org/abs/2402.04216) | 通过资源感知的分层联邦学习，我们提出了一种解决方案，可以预测用户未来的内容请求，并减轻无线视频缓存网络中回程流量拥塞的问题。 |
| [^2] | [Optimal Control of Logically Constrained Partially Observable and Multi-Agent Markov Decision Processes.](http://arxiv.org/abs/2305.14736) | 本文介绍了一个用于部分可观察和多智能体马尔可夫决策过程的最优控制理论，能够使用时间逻辑规范表达约束，并提供了一种结构化的方法来合成策略以最大化累积奖励并保证约束条件的概率足够高。同时我们还提供了对信息不对称的多智能体设置进行最优控制的框架。 |

# 详细

[^1]: 无线视频缓存网络中的资源感知分层联邦学习

    Resource-Aware Hierarchical Federated Learning in Wireless Video Caching Networks

    [https://arxiv.org/abs/2402.04216](https://arxiv.org/abs/2402.04216)

    通过资源感知的分层联邦学习，我们提出了一种解决方案，可以预测用户未来的内容请求，并减轻无线视频缓存网络中回程流量拥塞的问题。

    

    在无线视频缓存网络中，通过将待请求内容存储在不同级别上，可以减轻由少数热门文件的视频流量造成的回程拥塞。通常，内容服务提供商（CSP）拥有内容，用户使用其（无线）互联网服务提供商（ISP）从CSP请求其首选内容。由于这些参与方不会透露其私密信息和商业机密，传统技术可能无法用于预测用户未来需求的动态变化。出于这个原因，我们提出了一种新颖的资源感知分层联邦学习（RawHFL）解决方案，用于预测用户未来的内容请求。采用了一种实用的数据获取技术，允许用户根据其请求的内容更新其本地训练数据集。此外，由于网络和其他计算资源有限，考虑到只有一部分用户参与模型训练，我们推导出

    Backhaul traffic congestion caused by the video traffic of a few popular files can be alleviated by storing the to-be-requested content at various levels in wireless video caching networks. Typically, content service providers (CSPs) own the content, and the users request their preferred content from the CSPs using their (wireless) internet service providers (ISPs). As these parties do not reveal their private information and business secrets, traditional techniques may not be readily used to predict the dynamic changes in users' future demands. Motivated by this, we propose a novel resource-aware hierarchical federated learning (RawHFL) solution for predicting user's future content requests. A practical data acquisition technique is used that allows the user to update its local training dataset based on its requested content. Besides, since networking and other computational resources are limited, considering that only a subset of the users participate in the model training, we derive
    
[^2]: 逻辑约束下的部分可观察和多智能体马尔可夫决策过程的最优控制

    Optimal Control of Logically Constrained Partially Observable and Multi-Agent Markov Decision Processes. (arXiv:2305.14736v1 [cs.AI])

    [http://arxiv.org/abs/2305.14736](http://arxiv.org/abs/2305.14736)

    本文介绍了一个用于部分可观察和多智能体马尔可夫决策过程的最优控制理论，能够使用时间逻辑规范表达约束，并提供了一种结构化的方法来合成策略以最大化累积奖励并保证约束条件的概率足够高。同时我们还提供了对信息不对称的多智能体设置进行最优控制的框架。

    

    自动化系统通常会产生逻辑约束，例如来自安全、操作或法规要求，可以用时间逻辑规范表达这些约束。系统状态通常是部分可观察的，可能包含具有共同目标但不同信息结构和约束的多个智能体。在本文中，我们首先引入了一个最优控制理论，用于具有有限线性时间逻辑约束的部分可观察马尔可夫决策过程（POMDP）。我们提供了一种结构化方法，用于合成策略，同时确保满足时间逻辑约束的概率足够高时最大化累积回报。我们的方法具有关于近似奖励最优性和约束满足的保证。然后我们在此基础上构建了一个对信息不对称的具有逻辑约束的多智能体设置进行最优控制的框架。我们阐述了该方法并给出了理论保证。

    Autonomous systems often have logical constraints arising, for example, from safety, operational, or regulatory requirements. Such constraints can be expressed using temporal logic specifications. The system state is often partially observable. Moreover, it could encompass a team of multiple agents with a common objective but disparate information structures and constraints. In this paper, we first introduce an optimal control theory for partially observable Markov decision processes (POMDPs) with finite linear temporal logic constraints. We provide a structured methodology for synthesizing policies that maximize a cumulative reward while ensuring that the probability of satisfying a temporal logic constraint is sufficiently high. Our approach comes with guarantees on approximate reward optimality and constraint satisfaction. We then build on this approach to design an optimal control framework for logically constrained multi-agent settings with information asymmetry. We illustrate the
    

