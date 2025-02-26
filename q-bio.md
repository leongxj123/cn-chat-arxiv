# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enhancing selectivity using Wasserstein distance based reweighing.](http://arxiv.org/abs/2401.11562) | 我们设计了一种使用Wasserstein距离进行加权的算法，在标记的数据集上训练神经网络可以逼近在其他数据集上训练得到的结果。我们证明了算法可以输出接近最优的加权，且算法简单可扩展。我们的算法可以有意地引入分布偏移进行多目标优化。作为应用实例，我们训练了一个神经网络来识别对细胞信号传导的MAP激酶具有非结合性的小分子结合物。 |

# 详细

[^1]: 使用Wasserstein距离进行加权以增强选择性

    Enhancing selectivity using Wasserstein distance based reweighing. (arXiv:2401.11562v1 [stat.ML])

    [http://arxiv.org/abs/2401.11562](http://arxiv.org/abs/2401.11562)

    我们设计了一种使用Wasserstein距离进行加权的算法，在标记的数据集上训练神经网络可以逼近在其他数据集上训练得到的结果。我们证明了算法可以输出接近最优的加权，且算法简单可扩展。我们的算法可以有意地引入分布偏移进行多目标优化。作为应用实例，我们训练了一个神经网络来识别对细胞信号传导的MAP激酶具有非结合性的小分子结合物。

    

    给定两个标记数据集𝒮和𝒯，我们设计了一种简单高效的贪婪算法来对损失函数进行加权，使得在𝒮上训练得到的神经网络权重的极限分布逼近在𝒯上训练得到的极限分布。在理论方面，我们证明了当输入数据集的度量熵有界时，我们的贪婪算法输出接近最优的加权，即网络权重的两个不变分布在总变差距离上可以证明接近。此外，该算法简单可扩展，并且我们还证明了算法的效率上界。我们的算法可以有意地引入分布偏移以进行（软）多目标优化。作为一个动机应用，我们训练了一个神经网络来识别对MNK2（一种细胞信号传导的MAP激酶）具有非结合性的小分子结合物。

    Given two labeled data-sets $\mathcal{S}$ and $\mathcal{T}$, we design a simple and efficient greedy algorithm to reweigh the loss function such that the limiting distribution of the neural network weights that result from training on $\mathcal{S}$ approaches the limiting distribution that would have resulted by training on $\mathcal{T}$.  On the theoretical side, we prove that when the metric entropy of the input data-sets is bounded, our greedy algorithm outputs a close to optimal reweighing, i.e., the two invariant distributions of network weights will be provably close in total variation distance. Moreover, the algorithm is simple and scalable, and we prove bounds on the efficiency of the algorithm as well.  Our algorithm can deliberately introduce distribution shift to perform (soft) multi-criteria optimization. As a motivating application, we train a neural net to recognize small molecule binders to MNK2 (a MAP Kinase, responsible for cell signaling) which are non-binders to MNK1
    

