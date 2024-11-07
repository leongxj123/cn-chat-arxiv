# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Thresholded Oja does Sparse PCA?](https://arxiv.org/abs/2402.07240) | 阈值和重新归一化Oja算法的输出可获得一个接近最优的错误率，与未经阈值处理的Oja向量相比，这大大减小了误差。 |
| [^2] | [Keep or toss? A nonparametric score to evaluate solutions for noisy ICA](https://arxiv.org/abs/2401.08468) | 本文提出一种非参数分数来自适应选择适用于任意高斯噪声的ICA算法，并通过特征函数评估估计的混合矩阵质量，无需了解噪声分布参数。 |
| [^3] | [Mind the spikes: Benign overfitting of kernels and neural networks in fixed dimension.](http://arxiv.org/abs/2305.14077) | 这篇论文研究了固定维度下内核和神经网络的良性过拟合，发现良性过拟合的关键在于估计器的平滑度而不是维数，并证明在固定维度下中度导数的良性过拟合是不可能的。相反，我们证明了用序列核进行回归是可能出现良性过拟合的。 |

# 详细

[^1]: 阈值Oja是否适用于稀疏PCA？

    Thresholded Oja does Sparse PCA?

    [https://arxiv.org/abs/2402.07240](https://arxiv.org/abs/2402.07240)

    阈值和重新归一化Oja算法的输出可获得一个接近最优的错误率，与未经阈值处理的Oja向量相比，这大大减小了误差。

    

    我们考虑了当比值$d/n \rightarrow c > 0$时稀疏主成分分析（PCA）的问题。在离线设置下，关于稀疏PCA的最优率已经有很多研究，其中所有数据都可以用于多次传递。相比之下，当人口特征向量是$s$-稀疏时，具有$O(d)$存储和$O(nd)$时间复杂度的流算法通常要求强初始化条件，否则会有次优错误。我们展示了一种简单的算法，对Oja算法的输出（Oja向量）进行阈值和重新归一化，从而获得接近最优的错误率。这非常令人惊讶，因为没有阈值，Oja向量的误差很大。我们的分析集中在限制未归一化的Oja向量的项上，这涉及将一组独立随机矩阵的乘积在随机初始向量上的投影。 这是非平凡且新颖的，因为以前的Oja算法分析没有考虑这一点。

    arXiv:2402.07240v2 Announce Type: cross  Abstract: We consider the problem of Sparse Principal Component Analysis (PCA) when the ratio $d/n \rightarrow c > 0$. There has been a lot of work on optimal rates on sparse PCA in the offline setting, where all the data is available for multiple passes. In contrast, when the population eigenvector is $s$-sparse, streaming algorithms that have $O(d)$ storage and $O(nd)$ time complexity either typically require strong initialization conditions or have a suboptimal error. We show that a simple algorithm that thresholds and renormalizes the output of Oja's algorithm (the Oja vector) obtains a near-optimal error rate. This is very surprising because, without thresholding, the Oja vector has a large error. Our analysis centers around bounding the entries of the unnormalized Oja vector, which involves the projection of a product of independent random matrices on a random initial vector. This is nontrivial and novel since previous analyses of Oja's al
    
[^2]: 保留还是丢弃？一种评估有噪声ICA解决方案的非参数分数

    Keep or toss? A nonparametric score to evaluate solutions for noisy ICA

    [https://arxiv.org/abs/2401.08468](https://arxiv.org/abs/2401.08468)

    本文提出一种非参数分数来自适应选择适用于任意高斯噪声的ICA算法，并通过特征函数评估估计的混合矩阵质量，无需了解噪声分布参数。

    

    独立分量分析（ICA）于20世纪80年代引入，作为盲源分离（BSS）的模型，指的是在对混合信号进行恢复时，对源信号或混合过程了解有限的情况下的过程。尽管有许多精密算法进行估计，但不同方法存在不同的缺点。在本文中，我们开发了一种非参数分数，用于自适应地选择ICA算法和任意高斯噪声。该分数的创新之处在于，它只假设数据具有有限的二阶矩，并使用特征函数来评估估计的混合矩阵的质量，而无需了解噪声分布的参数。此外，我们提出了一些新的对比函数和算法，它们具有与现有算法（如FASTICA和JADE）相同的快速计算性能，但在前者可能失败的领域中工作。尽管这些方法也可能存在缺点，

    Independent Component Analysis (ICA) was introduced in the 1980's as a model for Blind Source Separation (BSS), which refers to the process of recovering the sources underlying a mixture of signals, with little knowledge about the source signals or the mixing process. While there are many sophisticated algorithms for estimation, different methods have different shortcomings. In this paper, we develop a nonparametric score to adaptively pick the right algorithm for ICA with arbitrary Gaussian noise. The novelty of this score stems from the fact that it just assumes a finite second moment of the data and uses the characteristic function to evaluate the quality of the estimated mixing matrix without any knowledge of the parameters of the noise distribution. In addition, we propose some new contrast functions and algorithms that enjoy the same fast computability as existing algorithms like FASTICA and JADE but work in domains where the former may fail. While these also may have weaknesses,
    
[^3]: 警惕尖峰：固定维度下内核和神经网络的良性过拟合

    Mind the spikes: Benign overfitting of kernels and neural networks in fixed dimension. (arXiv:2305.14077v1 [stat.ML])

    [http://arxiv.org/abs/2305.14077](http://arxiv.org/abs/2305.14077)

    这篇论文研究了固定维度下内核和神经网络的良性过拟合，发现良性过拟合的关键在于估计器的平滑度而不是维数，并证明在固定维度下中度导数的良性过拟合是不可能的。相反，我们证明了用序列核进行回归是可能出现良性过拟合的。

    

    过度参数化的神经网络训练达到接近零的训练误差的成功引起了人们对良性过拟合现象的极大兴趣，即使估计器插值嘈杂的训练数据，它们还是具有统计一致性。尽管某些学习方法的固定维度下已经确定了良性过拟合，但目前的文献表明，对于典型内核方法和宽神经网络的回归，良性过拟合需要高维度设置，其中维数随着样本大小的增加而增加。本文表明，估计器的平滑度是关键，而不是维数：只有当估计器的导数足够大时，良性过拟合才可能发生。我们将现有的不一致性结果推广到非插值模型和更多内核，以表明在固定维度下中度导数的良性过拟合是不可能的。相反，我们证明了用序列核进行回归是可能出现良性过拟合的。

    The success of over-parameterized neural networks trained to near-zero training error has caused great interest in the phenomenon of benign overfitting, where estimators are statistically consistent even though they interpolate noisy training data. While benign overfitting in fixed dimension has been established for some learning methods, current literature suggests that for regression with typical kernel methods and wide neural networks, benign overfitting requires a high-dimensional setting where the dimension grows with the sample size. In this paper, we show that the smoothness of the estimators, and not the dimension, is the key: benign overfitting is possible if and only if the estimator's derivatives are large enough. We generalize existing inconsistency results to non-interpolating models and more kernels to show that benign overfitting with moderate derivatives is impossible in fixed dimension. Conversely, we show that benign overfitting is possible for regression with a seque
    

