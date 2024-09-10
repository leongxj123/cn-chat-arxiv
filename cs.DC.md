# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Differentially Private Online Federated Learning with Correlated Noise](https://arxiv.org/abs/2403.16542) | 提出一种利用相关噪声提高效用并确保隐私的差分隐私在线联邦学习算法，解决了DP噪声和本地更新带来的挑战，并在动态环境中建立了动态遗憾界。 |
| [^2] | [Cooperative Minibatching in Graph Neural Networks.](http://arxiv.org/abs/2310.12403) | 本文提出了一种协作小批处理的方法来解决图神经网络中的邻域爆炸现象（NEP），该方法通过利用采样子图的大小与批处理大小的关系来减少每个种子顶点的工作量。 |

# 详细

[^1]: 具有相关噪声的差分隐私在线联邦学习

    Differentially Private Online Federated Learning with Correlated Noise

    [https://arxiv.org/abs/2403.16542](https://arxiv.org/abs/2403.16542)

    提出一种利用相关噪声提高效用并确保隐私的差分隐私在线联邦学习算法，解决了DP噪声和本地更新带来的挑战，并在动态环境中建立了动态遗憾界。

    

    我们提出了一种新颖的差分隐私算法，用于在线联邦学习，利用时间相关的噪声来提高效用同时确保连续发布的模型的隐私性。为了解决源自DP噪声和本地更新带来的流式非独立同分布数据的挑战，我们开发了扰动迭代分析来控制DP噪声对效用的影响。此外，我们展示了在准强凸条件下如何有效管理来自本地更新的漂移误差。在$(\epsilon, \delta)$-DP预算范围内，我们建立了整个时间段上的动态遗憾界，量化了关键参数的影响以及动态环境变化的强度。数值实验证实了所提算法的有效性。

    arXiv:2403.16542v1 Announce Type: new  Abstract: We propose a novel differentially private algorithm for online federated learning that employs temporally correlated noise to improve the utility while ensuring the privacy of the continuously released models. To address challenges stemming from DP noise and local updates with streaming noniid data, we develop a perturbed iterate analysis to control the impact of the DP noise on the utility. Moreover, we demonstrate how the drift errors from local updates can be effectively managed under a quasi-strong convexity condition. Subject to an $(\epsilon, \delta)$-DP budget, we establish a dynamic regret bound over the entire time horizon that quantifies the impact of key parameters and the intensity of changes in dynamic environments. Numerical experiments validate the efficacy of the proposed algorithm.
    
[^2]: 图神经网络中的协作小批次

    Cooperative Minibatching in Graph Neural Networks. (arXiv:2310.12403v1 [cs.LG])

    [http://arxiv.org/abs/2310.12403](http://arxiv.org/abs/2310.12403)

    本文提出了一种协作小批处理的方法来解决图神经网络中的邻域爆炸现象（NEP），该方法通过利用采样子图的大小与批处理大小的关系来减少每个种子顶点的工作量。

    

    在大规模训练图神经网络（GNN）时需要大量的计算资源，这个过程非常密集。减少资源需求的最有效方法之一是将小批量训练与图采样相结合。GNN具有一个独特的特性，即小批量中的项具有重叠的数据。然而，常用的独立小批量方法将每个处理单元（PE）分配给自己的小批量进行处理，导致重复计算和跨PE的输入数据访问。这放大了邻域爆炸现象（NEP），这是限制扩展性的主要瓶颈。为了减少多PE环境中NEP的影响，我们提出了一种新的方法，称为协作小批处理。我们的方法利用采样子图的大小是批处理大小的凹函数这一特性，可以明显减少每个种子顶点的工作量，同时增加批处理大小。因此，这是一种有利的方法。

    Significant computational resources are required to train Graph Neural Networks (GNNs) at a large scale, and the process is highly data-intensive. One of the most effective ways to reduce resource requirements is minibatch training coupled with graph sampling. GNNs have the unique property that items in a minibatch have overlapping data. However, the commonly implemented Independent Minibatching approach assigns each Processing Element (PE) its own minibatch to process, leading to duplicated computations and input data access across PEs. This amplifies the Neighborhood Explosion Phenomenon (NEP), which is the main bottleneck limiting scaling. To reduce the effects of NEP in the multi-PE setting, we propose a new approach called Cooperative Minibatching. Our approach capitalizes on the fact that the size of the sampled subgraph is a concave function of the batch size, leading to significant reductions in the amount of work per seed vertex as batch sizes increase. Hence, it is favorable 
    

