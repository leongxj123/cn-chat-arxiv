# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fast Convergence Federated Learning with Aggregated Gradients.](http://arxiv.org/abs/2303.15799) | 该论文提出了一种带有聚合梯度的快速收敛联邦学习方法，通过引入均值场方法来完成参数和梯度的聚合步骤，该方法在收敛速度和通信成本方面优于传统方法。 |

# 详细

[^1]: 带有聚合梯度的快速收敛联邦学习

    Fast Convergence Federated Learning with Aggregated Gradients. (arXiv:2303.15799v1 [cs.LG])

    [http://arxiv.org/abs/2303.15799](http://arxiv.org/abs/2303.15799)

    该论文提出了一种带有聚合梯度的快速收敛联邦学习方法，通过引入均值场方法来完成参数和梯度的聚合步骤，该方法在收敛速度和通信成本方面优于传统方法。

    

    联邦学习（FL）是一种新的机器学习框架，它使多个分布式设备在保护本地数据的同时，通过中央服务器协同训练共享模型。然而，非独立和同分布（Non-IID）的数据样本以及参与者之间频繁的通信将减缓收敛速率并增加通信成本。为了实现快速收敛，我们通过在常规本地更新规则中引入聚合梯度来改善本地梯度下降方法，并提出一种自适应学习率算法，在每次迭代中进一步考虑本地参数和全局参数的偏差。以上策略要求在每个本地迭代中收集所有客户端的本地参数和梯度，由于本地更新期间没有通信，这是具有挑战性的。因此，我们利用均值场方法，引入称为全局均值场和本地均值场的两个均值场术语来完成聚合步骤。实验结果表明，我们提出的方法在收敛速度和通信成本方面优于传统方法。

    Federated Learning (FL) is a novel machine learning framework, which enables multiple distributed devices cooperatively training a shared model scheduled by a central server while protecting private data locally. However, the non-independent-and-identically-distributed (Non-IID) data samples and frequent communication among participants will slow down the convergent rate and increase communication costs. To achieve fast convergence, we ameliorate the local gradient descend approach in conventional local update rule by introducing the aggregated gradients at each local update epoch, and propose an adaptive learning rate algorithm that further takes the deviation of local parameter and global parameter into consideration at each iteration. The above strategy requires all clients' local parameters and gradients at each local iteration, which is challenging as there is no communication during local update epochs. Accordingly, we utilize mean field approach by introducing two mean field ter
    

