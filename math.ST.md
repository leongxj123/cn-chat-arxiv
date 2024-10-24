# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Online Differentially Private Synthetic Data Generation](https://arxiv.org/abs/2402.08012) | 本文提出了一种在线差分隐私合成数据生成的多项式时间算法，在超立方体数据流上实现了近乎最优的精度界限，也推广了之前关于计数查询的连续发布模型的工作，仅需要额外的多项式对数因子。 |
| [^2] | [A general theory for robust clustering via trimmed mean.](http://arxiv.org/abs/2401.05574) | 本文提出了一种通过使用修剪均值类型的中心点估计的混合聚类技术，用于在存在次高斯误差的中心点周围分布的弱初始化条件下产生最优错误标记保证，并且在存在敌对异常值的情况下仍然有效。 |
| [^3] | [Bayes optimal learning in high-dimensional linear regression with network side information.](http://arxiv.org/abs/2306.05679) | 本文首次研究了具有网络辅助信息的高维线性回归中的贝叶斯最优学习问题，引入了Reg-Graph模型并提出了基于AMP的迭代算法，在实验中优于现有的几种网络辅助回归方法。 |

# 详细

[^1]: 在线差分隐私合成数据生成

    Online Differentially Private Synthetic Data Generation

    [https://arxiv.org/abs/2402.08012](https://arxiv.org/abs/2402.08012)

    本文提出了一种在线差分隐私合成数据生成的多项式时间算法，在超立方体数据流上实现了近乎最优的精度界限，也推广了之前关于计数查询的连续发布模型的工作，仅需要额外的多项式对数因子。

    

    本文提出了一种用于在线差分隐私合成数据生成的多项式时间算法。对于在超立方体$[0,1]^d$内的数据流和无限时间范围，我们开发了一种在线算法，每个时间$t$都生成一个差分隐私合成数据集。该算法在1-Wasserstein距离上实现了近乎最优的精度界限：当$d\geq 2$时为$O(t^{-1/d}\log(t)$，当$d=1$时为$O(t^{-1}\log^{4.5}(t)$。这个结果将之前关于计数查询的连续发布模型的工作推广到包括Lipschitz查询。与离线情况不同，离线情况下整个数据集一次性可用，我们的方法仅需要在精度界限中额外的多项式对数因子。

    We present a polynomial-time algorithm for online differentially private synthetic data generation. For a data stream within the hypercube $[0,1]^d$ and an infinite time horizon, we develop an online algorithm that generates a differentially private synthetic dataset at each time $t$. This algorithm achieves a near-optimal accuracy bound of $O(t^{-1/d}\log(t))$ for $d\geq 2$ and $O(t^{-1}\log^{4.5}(t))$ for $d=1$ in the 1-Wasserstein distance. This result generalizes the previous work on the continual release model for counting queries to include Lipschitz queries. Compared to the offline case, where the entire dataset is available at once, our approach requires only an extra polylog factor in the accuracy bound.
    
[^2]: 通过修剪均值的鲁棒聚类的一般理论

    A general theory for robust clustering via trimmed mean. (arXiv:2401.05574v1 [math.ST])

    [http://arxiv.org/abs/2401.05574](http://arxiv.org/abs/2401.05574)

    本文提出了一种通过使用修剪均值类型的中心点估计的混合聚类技术，用于在存在次高斯误差的中心点周围分布的弱初始化条件下产生最优错误标记保证，并且在存在敌对异常值的情况下仍然有效。

    

    在存在异质数据的统计机器学习中，聚类是一种基本工具。许多最近的结果主要关注在数据围绕带有次高斯误差的中心点分布时的最优错误标记保证。然而，限制性的次高斯模型在实践中常常无效，因为各种实际应用展示了围绕中心点的重尾分布或受到可能的敌对攻击，需要具有鲁棒数据驱动初始化的鲁棒聚类。在本文中，我们引入一种混合聚类技术，利用一种新颖的多变量修剪均值类型的中心点估计，在中心点周围的误差分布的弱初始化条件下产生错误标记保证。我们还给出了一个相匹配的下界，上界依赖于聚类的数量。此外，我们的方法即使在存在敌对异常值的情况下也能产生最优错误标记。我们的结果简化为亚高斯模型的情况。

    Clustering is a fundamental tool in statistical machine learning in the presence of heterogeneous data. Many recent results focus primarily on optimal mislabeling guarantees, when data are distributed around centroids with sub-Gaussian errors. Yet, the restrictive sub-Gaussian model is often invalid in practice, since various real-world applications exhibit heavy tail distributions around the centroids or suffer from possible adversarial attacks that call for robust clustering with a robust data-driven initialization. In this paper, we introduce a hybrid clustering technique with a novel multivariate trimmed mean type centroid estimate to produce mislabeling guarantees under a weak initialization condition for general error distributions around the centroids. A matching lower bound is derived, up to factors depending on the number of clusters. In addition, our approach also produces the optimal mislabeling even in the presence of adversarial outliers. Our results reduce to the sub-Gaus
    
[^3]: 具有网络辅助信息的高维线性回归中的贝叶斯最优学习

    Bayes optimal learning in high-dimensional linear regression with network side information. (arXiv:2306.05679v1 [math.ST])

    [http://arxiv.org/abs/2306.05679](http://arxiv.org/abs/2306.05679)

    本文首次研究了具有网络辅助信息的高维线性回归中的贝叶斯最优学习问题，引入了Reg-Graph模型并提出了基于AMP的迭代算法，在实验中优于现有的几种网络辅助回归方法。

    

    在基因组学、蛋白质组学和神经科学等应用中，具有网络辅助信息的监督学习问题经常出现。本文中，我们首次研究了具有网络辅助信息的高维线性回归中的贝叶斯最优学习问题。为此，我们首先引入了一个简单的生成模型（称为Reg-Graph模型），通过一组共同的潜在参数为监督数据和观测到的网络设定了一个联合分布。接下来，我们介绍了一种基于近似消息传递（AMP）的迭代算法，在非常一般的条件下可证明是贝叶斯最优的。此外，我们对潜在信号和观测到的数据之间的极限互信息进行了表征，从而精确量化了网络辅助信息在回归问题中的统计影响。我们对模拟数据和实际数据的实验表明，我们的方法优于现有的几种网络辅助回归方法。

    Supervised learning problems with side information in the form of a network arise frequently in applications in genomics, proteomics and neuroscience. For example, in genetic applications, the network side information can accurately capture background biological information on the intricate relations among the relevant genes. In this paper, we initiate a study of Bayes optimal learning in high-dimensional linear regression with network side information. To this end, we first introduce a simple generative model (called the Reg-Graph model) which posits a joint distribution for the supervised data and the observed network through a common set of latent parameters. Next, we introduce an iterative algorithm based on Approximate Message Passing (AMP) which is provably Bayes optimal under very general conditions. In addition, we characterize the limiting mutual information between the latent signal and the data observed, and thus precisely quantify the statistical impact of the network side 
    

