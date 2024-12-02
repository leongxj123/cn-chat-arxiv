# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Frequency and cardinality recovery from sketched data: a novel approach bridging Bayesian and frequentist views.](http://arxiv.org/abs/2309.15408) | 该论文研究了如何仅使用压缩表示来恢复大规模数据集中符号的频率，并引入了新的估计方法，将贝叶斯和频率论观点结合起来，提供了更好的解决方案。此外，还扩展了该方法以解决基数恢复问题。 |
| [^2] | [On Consistency of Signatures Using Lasso.](http://arxiv.org/abs/2305.10413) | 本文重新审视了Lasso回归对于签名变换的一致性问题，并发现对于不同的过程和时间序列，选择适当的签名定义和随机模型可以提高Lasso回归的一致性。 |

# 详细

[^1]: 从压缩数据中恢复频率和基数：一种将贝叶斯和频率论观点连接起来的新方法

    Frequency and cardinality recovery from sketched data: a novel approach bridging Bayesian and frequentist views. (arXiv:2309.15408v1 [stat.ME])

    [http://arxiv.org/abs/2309.15408](http://arxiv.org/abs/2309.15408)

    该论文研究了如何仅使用压缩表示来恢复大规模数据集中符号的频率，并引入了新的估计方法，将贝叶斯和频率论观点结合起来，提供了更好的解决方案。此外，还扩展了该方法以解决基数恢复问题。

    

    我们研究如何仅使用通过随机哈希获得的对数据进行压缩表示或草图来恢复大规模离散数据集中符号的频率。这是一个在计算机科学中的经典问题，有各种算法可用，如计数最小草图。然而，这些算法通常假设数据是固定的，处理随机采样数据时估计过于保守且可能不准确。在本文中，我们将草图数据视为未知分布的随机样本，然后引入改进现有方法的新估计器。我们的方法结合了贝叶斯非参数和经典（频率论）观点，解决了它们独特的限制，提供了一个有原则且实用的解决方案。此外，我们扩展了我们的方法以解决相关但不同的基数恢复问题，该问题涉及估计数据集中不同对象的总数。

    We study how to recover the frequency of a symbol in a large discrete data set, using only a compressed representation, or sketch, of those data obtained via random hashing. This is a classical problem in computer science, with various algorithms available, such as the count-min sketch. However, these algorithms often assume that the data are fixed, leading to overly conservative and potentially inaccurate estimates when dealing with randomly sampled data. In this paper, we consider the sketched data as a random sample from an unknown distribution, and then we introduce novel estimators that improve upon existing approaches. Our method combines Bayesian nonparametric and classical (frequentist) perspectives, addressing their unique limitations to provide a principled and practical solution. Additionally, we extend our method to address the related but distinct problem of cardinality recovery, which consists of estimating the total number of distinct objects in the data set. We validate
    
[^2]: 使用Lasso的签名一致性研究

    On Consistency of Signatures Using Lasso. (arXiv:2305.10413v1 [stat.ML])

    [http://arxiv.org/abs/2305.10413](http://arxiv.org/abs/2305.10413)

    本文重新审视了Lasso回归对于签名变换的一致性问题，并发现对于不同的过程和时间序列，选择适当的签名定义和随机模型可以提高Lasso回归的一致性。

    

    签名变换是连续和离散时间序列数据的迭代路径积分，它们的普遍非线性通过线性化特征选择问题。本文在理论和数值上重新审视了Lasso回归对于签名变换的一致性问题。我们的研究表明，对于更接近布朗运动或具有较弱跨维度相关性的过程和时间序列，签名定义为It\^o积分的Lasso回归更具一致性；对于均值回归过程和时间序列，其签名定义为Stratonovich积分在Lasso回归中具有更高的一致性。我们的发现强调了在统计推断和机器学习中选择适当的签名和随机模型的重要性。

    Signature transforms are iterated path integrals of continuous and discrete-time time series data, and their universal nonlinearity linearizes the problem of feature selection. This paper revisits the consistency issue of Lasso regression for the signature transform, both theoretically and numerically. Our study shows that, for processes and time series that are closer to Brownian motion or random walk with weaker inter-dimensional correlations, the Lasso regression is more consistent for their signatures defined by It\^o integrals; for mean reverting processes and time series, their signatures defined by Stratonovich integrals have more consistency in the Lasso regression. Our findings highlight the importance of choosing appropriate definitions of signatures and stochastic models in statistical inference and machine learning.
    

