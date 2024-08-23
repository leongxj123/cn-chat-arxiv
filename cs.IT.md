# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Beyond PCA: A Probabilistic Gram-Schmidt Approach to Feature Extraction](https://arxiv.org/abs/2311.09386) | 本研究提出了一种概率性Gram-Schmidt方法来进行特征提取，该方法可以检测和去除非线性依赖性，从而提取数据中的线性特征并去除非线性冗余。 |
| [^2] | [Leveraging Variational Autoencoders for Parameterized MMSE Channel Estimation.](http://arxiv.org/abs/2307.05352) | 本文提出利用变分自编码器进行信道估计，通过对条件高斯信道模型的内部结构进行参数化逼近来获得均方根误差最优信道估计器，同时给出了基于变分自编码器的估计器的实用性考虑和三种不同训练方式的估计器变体。 |

# 详细

[^1]: 超越PCA：一种概率性Gram-Schmidt方法的特征提取

    Beyond PCA: A Probabilistic Gram-Schmidt Approach to Feature Extraction

    [https://arxiv.org/abs/2311.09386](https://arxiv.org/abs/2311.09386)

    本研究提出了一种概率性Gram-Schmidt方法来进行特征提取，该方法可以检测和去除非线性依赖性，从而提取数据中的线性特征并去除非线性冗余。

    

    在无监督学习中，线性特征提取在数据中存在非线性依赖的情况下是一个基本挑战。我们提出使用概率性Gram-Schmidt (GS)类型的正交化过程来检测和映射出冗余维度。具体而言，通过在一族函数上应用GS过程，该族函数预计捕捉到数据中的非线性依赖性，我们构建了一系列协方差矩阵，可以用于识别新的大方差方向，或者将这些依赖性从主成分中去除。在前一种情况下，我们提供了熵减少的信息理论保证。在后一种情况下，我们证明在某些假设下，所得算法在所选择函数族的线性张成空间中可以检测和去除非线性依赖性。两种提出的方法都可以从数据中提取线性特征并去除非线性冗余。

    Linear feature extraction at the presence of nonlinear dependencies among the data is a fundamental challenge in unsupervised learning. We propose using a probabilistic Gram-Schmidt (GS) type orthogonalization process in order to detect and map out redundant dimensions. Specifically, by applying the GS process over a family of functions which presumably captures the nonlinear dependencies in the data, we construct a series of covariance matrices that can either be used to identify new large-variance directions, or to remove those dependencies from the principal components. In the former case, we provide information-theoretic guarantees in terms of entropy reduction. In the latter, we prove that under certain assumptions the resulting algorithms detect and remove nonlinear dependencies whenever those dependencies lie in the linear span of the chosen function family. Both proposed methods extract linear features from the data while removing nonlinear redundancies. We provide simulation r
    
[^2]: 利用变分自编码器进行参数化MMSE信道估计

    Leveraging Variational Autoencoders for Parameterized MMSE Channel Estimation. (arXiv:2307.05352v1 [eess.SP])

    [http://arxiv.org/abs/2307.05352](http://arxiv.org/abs/2307.05352)

    本文提出利用变分自编码器进行信道估计，通过对条件高斯信道模型的内部结构进行参数化逼近来获得均方根误差最优信道估计器，同时给出了基于变分自编码器的估计器的实用性考虑和三种不同训练方式的估计器变体。

    

    在本文中，我们提出利用基于生成神经网络的变分自编码器进行信道估计。变分自编码器以一种新颖的方式将真实但未知的信道分布建模为条件高斯分布。所得到的信道估计器利用变分自编码器的内部结构对来自条件高斯信道模型的均方误差最优估计器进行参数化逼近。我们提供了严格的分析，以确定什么条件下基于变分自编码器的估计器是均方误差最优的。然后，我们提出了使基于变分自编码器的估计器实用的考虑因素，并提出了三种不同的估计器变体，它们在训练和评估阶段对信道知识的获取方式不同。特别地，仅基于噪声导频观测进行训练的所提出的估计器变体非常值得注意，因为它不需要获取信道训练。

    In this manuscript, we propose to utilize the generative neural network-based variational autoencoder for channel estimation. The variational autoencoder models the underlying true but unknown channel distribution as a conditional Gaussian distribution in a novel way. The derived channel estimator exploits the internal structure of the variational autoencoder to parameterize an approximation of the mean squared error optimal estimator resulting from the conditional Gaussian channel models. We provide a rigorous analysis under which conditions a variational autoencoder-based estimator is mean squared error optimal. We then present considerations that make the variational autoencoder-based estimator practical and propose three different estimator variants that differ in their access to channel knowledge during the training and evaluation phase. In particular, the proposed estimator variant trained solely on noisy pilot observations is particularly noteworthy as it does not require access
    

