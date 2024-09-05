# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fast and interpretable Support Vector Classification based on the truncated ANOVA decomposition](https://arxiv.org/abs/2402.02438) | 基于截断ANOVA分解的快速可解释支持向量分类法能够通过使用特征映射和少量维度的多变量基函数来快速且准确地进行高维散乱数据的分类。 |
| [^2] | [Adaptive operator learning for infinite-dimensional Bayesian inverse problems.](http://arxiv.org/abs/2310.17844) | 该论文提出了一种自适应操作员学习框架，通过使用贪婪算法选择自适应点对预训练的近似模型进行微调，逐渐减少建模误差。这种方法可以在准确性和效率之间取得平衡，有助于有效解决贝叶斯逆问题中的计算问题。 |

# 详细

[^1]: 基于截断ANOVA分解的快速可解释支持向量分类法

    Fast and interpretable Support Vector Classification based on the truncated ANOVA decomposition

    [https://arxiv.org/abs/2402.02438](https://arxiv.org/abs/2402.02438)

    基于截断ANOVA分解的快速可解释支持向量分类法能够通过使用特征映射和少量维度的多变量基函数来快速且准确地进行高维散乱数据的分类。

    

    支持向量机（SVM）是在散乱数据上进行分类的重要工具，在高维空间中通常需要处理许多数据点。我们提出使用基于三角函数或小波的特征映射来解决SVM的原始形式。在小维度设置中，快速傅里叶变换（FFT）和相关方法是处理所考虑基函数的强大工具。随着维度的增长，由于维数灾难，传统的基于FFT的方法变得低效。因此，我们限制自己使用多变量基函数，每个基函数只依赖于少数几个维度。这是由于效应的稀疏性和最近关于函数从散乱数据中的截断方差分解的重建的结果所带来的动机，使得生成的模型在特征的重要性以及它们的耦合方面具有可解释性。

    Support Vector Machines (SVMs) are an important tool for performing classification on scattered data, where one usually has to deal with many data points in high-dimensional spaces. We propose solving SVMs in primal form using feature maps based on trigonometric functions or wavelets. In small dimensional settings the Fast Fourier Transform (FFT) and related methods are a powerful tool in order to deal with the considered basis functions. For growing dimensions the classical FFT-based methods become inefficient due to the curse of dimensionality. Therefore, we restrict ourselves to multivariate basis functions, each one of them depends only on a small number of dimensions. This is motivated by the well-known sparsity of effects and recent results regarding the reconstruction of functions from scattered data in terms of truncated analysis of variance (ANOVA) decomposition, which makes the resulting model even interpretable in terms of importance of the features as well as their coupling
    
[^2]: 自适应操作员学习用于无限维贝叶斯逆问题

    Adaptive operator learning for infinite-dimensional Bayesian inverse problems. (arXiv:2310.17844v1 [math.NA])

    [http://arxiv.org/abs/2310.17844](http://arxiv.org/abs/2310.17844)

    该论文提出了一种自适应操作员学习框架，通过使用贪婪算法选择自适应点对预训练的近似模型进行微调，逐渐减少建模误差。这种方法可以在准确性和效率之间取得平衡，有助于有效解决贝叶斯逆问题中的计算问题。

    

    贝叶斯逆问题(BIPs)中的基本计算问题源于需要重复进行正向模型评估的要求。减少这种成本的一种常见策略是通过操作员学习使用计算效率高的近似方法替代昂贵的模型模拟，这受到了深度学习的最新进展的启发。然而，直接使用近似模型可能引入建模误差，加剧了逆问题已经存在的病态性。因此，在有效实施这些方法中，平衡准确性和效率至关重要。为此，我们开发了一个自适应操作员学习框架，可以通过强制在局部区域中准确拟合的代理逐渐减少建模误差。这是通过使用贪婪算法选择的自适应点在反演过程中对预训练的近似模型进行微调来实现的，该算法只需要少量的正向模型评估。

    The fundamental computational issues in Bayesian inverse problems (BIPs) governed by partial differential equations (PDEs) stem from the requirement of repeated forward model evaluations. A popular strategy to reduce such cost is to replace expensive model simulations by computationally efficient approximations using operator learning, motivated by recent progresses in deep learning. However, using the approximated model directly may introduce a modeling error, exacerbating the already ill-posedness of inverse problems. Thus, balancing between accuracy and efficiency is essential for the effective implementation of such approaches. To this end, we develop an adaptive operator learning framework that can reduce modeling error gradually by forcing the surrogate to be accurate in local areas. This is accomplished by fine-tuning the pre-trained approximate model during the inversion process with adaptive points selected by a greedy algorithm, which requires only a few forward model evaluat
    

