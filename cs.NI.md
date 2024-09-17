# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimized Network Architectures for Large Language Model Training with Billions of Parameters.](http://arxiv.org/abs/2307.12169) | 本文提出了一种优化的网络架构，用于训练拥有数十亿参数的大型语言模型。这个架构根据语言模型的通信需求，将集群分割成一组通过非阻塞高带宽互连的GPU集合，并通过轨道连接仅连接具有通信需求的GPU，从而降低网络成本高达75％，同时不影响训练性能。 |
| [^2] | [Dataset of Pathloss and ToA Radio Maps With Localization Application.](http://arxiv.org/abs/2212.11777) | 这个论文介绍了一个包含稠密城市环境中无线地图数据集的研究。这个数据集能够用于路径损耗预测和无线定位，通过在相同的城市地图上计算得到RSS和ToA地图，可以公平比较两种定位方法的效果。 |

# 详细

[^1]: 用于训练拥有数十亿参数的大型语言模型的优化网络架构

    Optimized Network Architectures for Large Language Model Training with Billions of Parameters. (arXiv:2307.12169v1 [cs.NI])

    [http://arxiv.org/abs/2307.12169](http://arxiv.org/abs/2307.12169)

    本文提出了一种优化的网络架构，用于训练拥有数十亿参数的大型语言模型。这个架构根据语言模型的通信需求，将集群分割成一组通过非阻塞高带宽互连的GPU集合，并通过轨道连接仅连接具有通信需求的GPU，从而降低网络成本高达75％，同时不影响训练性能。

    

    本文挑战了为训练大型语言模型（LLMs）构建任意到任意网络的传统范式。我们展示了LLMs呈现出一种独特的通信模式，在其中，只有小组的GPU需要高带宽的任意到任意通信，以实现接近最优的训练性能。在这些GPU小组之间，通信非常微不足道、稀疏且均匀。我们提出了一个新的网络架构，紧密匹配LLMs的通信需求。我们的架构将集群分割为一组通过非阻塞任意到任意高带宽互连的GPU集合，我们称之为HB域。在HB域之间，网络只连接具有通信需求的GPU。我们将这种网络连接称为“仅轨道连接”，并展示了我们的架构相对于最先进的任意到任意Clos网络可以将网络成本降低高达75％，同时不损害LLM训练的性能。

    This paper challenges the well-established paradigm for building any-to-any networks for training Large Language Models (LLMs). We show that LLMs exhibit a unique communication pattern where only small groups of GPUs require high-bandwidth any-to-any communication within them, to achieve near-optimal training performance. Across these groups of GPUs, the communication is insignificant, sparse, and homogeneous. We propose a new network architecture that closely resembles the communication requirement of LLMs. Our architecture partitions the cluster into sets of GPUs interconnected with non-blocking any-to-any high-bandwidth interconnects that we call HB domains. Across the HB domains, the network only connects GPUs with communication demands. We call this network a "rail-only" connection, and show that our proposed architecture reduces the network cost by up to 75% compared to the state-of-the-art any-to-any Clos networks without compromising the performance of LLM training.
    
[^2]: 具有定位应用的路径损耗和到达时间无线地图数据集

    Dataset of Pathloss and ToA Radio Maps With Localization Application. (arXiv:2212.11777v2 [cs.NI] UPDATED)

    [http://arxiv.org/abs/2212.11777](http://arxiv.org/abs/2212.11777)

    这个论文介绍了一个包含稠密城市环境中无线地图数据集的研究。这个数据集能够用于路径损耗预测和无线定位，通过在相同的城市地图上计算得到RSS和ToA地图，可以公平比较两种定位方法的效果。

    

    本文介绍了在稠密城市环境中生成并公开提供的一组无线地图数据集。这些数据集包括模拟的路径损耗/接收信号强度（RSS）和到达时间（ToA）无线地图，覆盖了大量真实城市地图的稠密城市设置。该数据集的两个主要应用是1）从输入的城市地图预测路径损耗的学习方法（即基于深度学习的模拟），以及2）无线定位。RSS和ToA地图通过相同的模拟在相同的城市地图上计算得出，可以对基于RSS和ToA的定位方法进行公平比较。

    In this article, we present a collection of radio map datasets in dense urban setting, which we generated and made publicly available. The datasets include simulated pathloss/received signal strength (RSS) and time of arrival (ToA) radio maps over a large collection of realistic dense urban setting in real city maps. The two main applications of the presented dataset are 1) learning methods that predict the pathloss from input city maps (namely, deep learning-based simulations), and, 2) wireless localization. The fact that the RSS and ToA maps are computed by the same simulations over the same city maps allows for a fair comparison of the RSS and ToA-based localization methods.
    

