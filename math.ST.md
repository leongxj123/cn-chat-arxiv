# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sampling from the Mean-Field Stationary Distribution](https://arxiv.org/abs/2402.07355) | 本文研究了从均场随机微分方程 (SDE) 的稳态分布中采样的复杂性，并提出了一种解耦的方法。该方法能够在多种情况下提供改进的保证，包括在均场区域优化某些双层神经网络的更好保证。 |
| [^2] | [A Survey on Statistical Theory of Deep Learning: Approximation, Training Dynamics, and Generative Models.](http://arxiv.org/abs/2401.07187) | 该论文综述了深度学习的统计理论，包括近似方法、训练动态和生成模型。在非参数框架中，结果揭示了神经网络过度风险的快速收敛速率，以及如何通过梯度方法训练网络以找到良好的泛化解决方案。 |
| [^3] | [Convergence of flow-based generative models via proximal gradient descent in Wasserstein space.](http://arxiv.org/abs/2310.17582) | 本文通过在Wasserstein空间中应用近端梯度下降，证明了基于流的生成模型的收敛性，并提供了生成数据分布的理论保证。 |
| [^4] | [Bootstrap-Assisted Inference for Generalized Grenander-type Estimators.](http://arxiv.org/abs/2303.13598) | 本文研究了广义Grenander型估计量的大样本分布特性，提出了Bootstrap-Aided推断方法，解决了标准非参数Bootstrap难以逼近广义Grenander型估计量的大样本分布的问题。 |
| [^5] | [High-dimensional variable clustering based on sub-asymptotic maxima of a weakly dependent random process.](http://arxiv.org/abs/2302.00934) | 我们提出了一种基于亚渐近极大值的高维变量聚类模型，该模型利用群集间多变量随机过程的极大值的独立性定义种群水平的群集，我们还开发了一种无需预先指定群集数量的算法来恢复变量的群集。该算法在特定条件下能够有效地识别数据中的群集，并能够以多项式复杂度进行计算。我们的工作对于理解依赖过程的块最大值的非参数学习有重要意义，并且在神经科学领域有着应用潜力。 |

# 详细

[^1]: 从均场稳态分布中采样

    Sampling from the Mean-Field Stationary Distribution

    [https://arxiv.org/abs/2402.07355](https://arxiv.org/abs/2402.07355)

    本文研究了从均场随机微分方程 (SDE) 的稳态分布中采样的复杂性，并提出了一种解耦的方法。该方法能够在多种情况下提供改进的保证，包括在均场区域优化某些双层神经网络的更好保证。

    

    我们研究了从均场随机微分方程 (SDE) 的稳态分布中采样的复杂性，或者等价地，即包含交互项的概率测度空间上的最小化函数的复杂性。我们的主要洞察是将这个问题的两个关键方面解耦：(1) 通过有限粒子系统逼近均场SDE，通过时间均匀传播混沌，和(2) 通过标准对数凹抽样器从有限粒子稳态分布中采样。我们的方法在概念上更简单，其灵活性允许结合用于算法和理论的最新技术。这导致在许多设置中提供了改进的保证，包括在均场区域优化某些双层神经网络的更好保证。

    We study the complexity of sampling from the stationary distribution of a mean-field SDE, or equivalently, the complexity of minimizing a functional over the space of probability measures which includes an interaction term.   Our main insight is to decouple the two key aspects of this problem: (1) approximation of the mean-field SDE via a finite-particle system, via uniform-in-time propagation of chaos, and (2) sampling from the finite-particle stationary distribution, via standard log-concave samplers. Our approach is conceptually simpler and its flexibility allows for incorporating the state-of-the-art for both algorithms and theory. This leads to improved guarantees in numerous settings, including better guarantees for optimizing certain two-layer neural networks in the mean-field regime.
    
[^2]: 深度学习的统计理论综述：近似，训练动态和生成模型

    A Survey on Statistical Theory of Deep Learning: Approximation, Training Dynamics, and Generative Models. (arXiv:2401.07187v1 [stat.ML])

    [http://arxiv.org/abs/2401.07187](http://arxiv.org/abs/2401.07187)

    该论文综述了深度学习的统计理论，包括近似方法、训练动态和生成模型。在非参数框架中，结果揭示了神经网络过度风险的快速收敛速率，以及如何通过梯度方法训练网络以找到良好的泛化解决方案。

    

    在这篇文章中，我们从三个角度回顾了关于神经网络统计理论的文献。第一部分回顾了在回归或分类的非参数框架下关于神经网络过度风险的结果。这些结果依赖于神经网络的显式构造，以及采用了近似理论的工具，导致过度风险的快速收敛速率。通过这些构造，可以用样本大小、数据维度和函数平滑性来表达网络的宽度和深度。然而，他们的基本分析仅适用于深度神经网络高度非凸的全局极小值点。这促使我们在第二部分回顾神经网络的训练动态。具体而言，我们回顾了那些试图回答“基于梯度方法训练的神经网络如何找到能够在未见数据上有良好泛化性能的解”的论文。尤其是两个知名的

    In this article, we review the literature on statistical theories of neural networks from three perspectives. In the first part, results on excess risks for neural networks are reviewed in the nonparametric framework of regression or classification. These results rely on explicit constructions of neural networks, leading to fast convergence rates of excess risks, in that tools from the approximation theory are adopted. Through these constructions, the width and depth of the networks can be expressed in terms of sample size, data dimension, and function smoothness. Nonetheless, their underlying analysis only applies to the global minimizer in the highly non-convex landscape of deep neural networks. This motivates us to review the training dynamics of neural networks in the second part. Specifically, we review papers that attempt to answer ``how the neural network trained via gradient-based methods finds the solution that can generalize well on unseen data.'' In particular, two well-know
    
[^3]: 在Wasserstein空间中通过近端梯度下降实现基于流的生成模型的收敛性

    Convergence of flow-based generative models via proximal gradient descent in Wasserstein space. (arXiv:2310.17582v1 [stat.ML])

    [http://arxiv.org/abs/2310.17582](http://arxiv.org/abs/2310.17582)

    本文通过在Wasserstein空间中应用近端梯度下降，证明了基于流的生成模型的收敛性，并提供了生成数据分布的理论保证。

    

    基于流的生成模型在计算数据生成和似然函数方面具有一定的优势，并且最近在实证表现上显示出竞争力。与相关基于分数扩散模型的积累理论研究相比，对于在正向（数据到噪声）和反向（噪声到数据）方向上都是确定性的流模型的分析还很少。本文通过在归一化流网络中实施Jordan-Kinderleherer-Otto（JKO）方案的所谓JKO流模型，提供了通过渐进流模型生成数据分布的理论保证。利用Wasserstein空间中近端梯度下降（GD）的指数收敛性，我们证明了通过JKO流模型生成数据的Kullback-Leibler（KL）保证为$O(\varepsilon^2)$，其中使用$N \lesssim \log (1/\varepsilon)$个JKO步骤（流中的$N$个残差块），其中$\varepsilon$是每步一阶条件的误差。

    Flow-based generative models enjoy certain advantages in computing the data generation and the likelihood, and have recently shown competitive empirical performance. Compared to the accumulating theoretical studies on related score-based diffusion models, analysis of flow-based models, which are deterministic in both forward (data-to-noise) and reverse (noise-to-data) directions, remain sparse. In this paper, we provide a theoretical guarantee of generating data distribution by a progressive flow model, the so-called JKO flow model, which implements the Jordan-Kinderleherer-Otto (JKO) scheme in a normalizing flow network. Leveraging the exponential convergence of the proximal gradient descent (GD) in Wasserstein space, we prove the Kullback-Leibler (KL) guarantee of data generation by a JKO flow model to be $O(\varepsilon^2)$ when using $N \lesssim \log (1/\varepsilon)$ many JKO steps ($N$ Residual Blocks in the flow) where $\varepsilon $ is the error in the per-step first-order condit
    
[^4]: 广义Grenander型估计量的Bootstrap-Aided推断

    Bootstrap-Assisted Inference for Generalized Grenander-type Estimators. (arXiv:2303.13598v1 [math.ST])

    [http://arxiv.org/abs/2303.13598](http://arxiv.org/abs/2303.13598)

    本文研究了广义Grenander型估计量的大样本分布特性，提出了Bootstrap-Aided推断方法，解决了标准非参数Bootstrap难以逼近广义Grenander型估计量的大样本分布的问题。

    

    Westling和Carone（2020）提出了一个框架来研究广义Grenander型估计量的大样本分布特性，这是一类用于单调函数的非参数估计器的多才多艺的类。这些估计量的极限分布可表示为高斯过程的最大凸支撑线的左导数，该高斯过程的协方差核可以很复杂，其单项式均值可以是未知阶数（如果感兴趣的函数的平坦度未知）。标准的非参数bootstrap即使知道均值的单项式顺序，也无法一致地逼近广义Grenander型估计量的大样本分布，这使得在应用中进行统计推断成为具有挑战性的任务。为了解决这个推断问题，我们提出了一种广义Grenander型估计量的bootstrap辅助推断程序。该程序依赖于一个精心设计但自动化的变换e

    Westling and Carone (2020) proposed a framework for studying the large sample distributional properties of generalized Grenander-type estimators, a versatile class of nonparametric estimators of monotone functions. The limiting distribution of those estimators is representable as the left derivative of the greatest convex minorant of a Gaussian process whose covariance kernel can be complicated and whose monomial mean can be of unknown order (when the degree of flatness of the function of interest is unknown). The standard nonparametric bootstrap is unable to consistently approximate the large sample distribution of the generalized Grenander-type estimators even if the monomial order of the mean is known, making statistical inference a challenging endeavour in applications. To address this inferential problem, we present a bootstrap-assisted inference procedure for generalized Grenander-type estimators. The procedure relies on a carefully crafted, yet automatic, transformation of the e
    
[^5]: 基于弱相关随机过程的亚渐近极大值的高维变量聚类

    High-dimensional variable clustering based on sub-asymptotic maxima of a weakly dependent random process. (arXiv:2302.00934v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2302.00934](http://arxiv.org/abs/2302.00934)

    我们提出了一种基于亚渐近极大值的高维变量聚类模型，该模型利用群集间多变量随机过程的极大值的独立性定义种群水平的群集，我们还开发了一种无需预先指定群集数量的算法来恢复变量的群集。该算法在特定条件下能够有效地识别数据中的群集，并能够以多项式复杂度进行计算。我们的工作对于理解依赖过程的块最大值的非参数学习有重要意义，并且在神经科学领域有着应用潜力。

    

    我们提出了一种新的变量聚类模型，称为渐近独立块 (AI-block) 模型，该模型基于群集间多变量平稳混合随机过程的极大值的独立性来定义种群水平的群集。该模型类是可识别的，意味着存在一种偏序关系，允许进行统计推断。我们还提出了一种算法，无需事先指定群集的数量即可恢复变量的群集。我们的工作提供了一些理论洞察，证明了在某些条件下，我们的算法能够在计算复杂性在维度中是多项式的情况下有效地识别数据中的群集。这意味着可以非参数地学习出仅仅是亚渐近的依赖过程的块最大值的群组。为了进一步说明我们的工作的重要性，我们将我们的方法应用于神经科学数据集。

    We propose a new class of models for variable clustering called Asymptotic Independent block (AI-block) models, which defines population-level clusters based on the independence of the maxima of a multivariate stationary mixing random process among clusters. This class of models is identifiable, meaning that there exists a maximal element with a partial order between partitions, allowing for statistical inference. We also present an algorithm for recovering the clusters of variables without specifying the number of clusters \emph{a priori}. Our work provides some theoretical insights into the consistency of our algorithm, demonstrating that under certain conditions it can effectively identify clusters in the data with a computational complexity that is polynomial in the dimension. This implies that groups can be learned nonparametrically in which block maxima of a dependent process are only sub-asymptotic. To further illustrate the significance of our work, we applied our method to neu
    

