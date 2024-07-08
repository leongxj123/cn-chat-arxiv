# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Gotta match 'em all: Solution diversification in graph matching matched filters.](http://arxiv.org/abs/2308.13451) | 本文提出了一种在大规模背景图中查找多个嵌入的模板图的新方法，通过迭代惩罚相似度矩阵来实现多样化匹配的发现，并提出了算法加速措施。在理论验证和实验证明中，证明了该方法的可行性和实用性。 |
| [^2] | [Engression: Extrapolation for Nonlinear Regression?.](http://arxiv.org/abs/2307.00835) | Engression是一种非线性回归方法，通过使用分布回归技术和预加性噪声模型，在训练样本范围边界外也能可靠地进行外推。 |
| [^3] | [Bootstrap-Assisted Inference for Generalized Grenander-type Estimators.](http://arxiv.org/abs/2303.13598) | 本文研究了广义Grenander型估计量的大样本分布特性，提出了Bootstrap-Aided推断方法，解决了标准非参数Bootstrap难以逼近广义Grenander型估计量的大样本分布的问题。 |
| [^4] | [High-dimensional variable clustering based on sub-asymptotic maxima of a weakly dependent random process.](http://arxiv.org/abs/2302.00934) | 我们提出了一种基于亚渐近极大值的高维变量聚类模型，该模型利用群集间多变量随机过程的极大值的独立性定义种群水平的群集，我们还开发了一种无需预先指定群集数量的算法来恢复变量的群集。该算法在特定条件下能够有效地识别数据中的群集，并能够以多项式复杂度进行计算。我们的工作对于理解依赖过程的块最大值的非参数学习有重要意义，并且在神经科学领域有着应用潜力。 |

# 详细

[^1]: 抓住它们：图匹配匹配滤波中的解决方案多样化

    Gotta match 'em all: Solution diversification in graph matching matched filters. (arXiv:2308.13451v1 [stat.ML])

    [http://arxiv.org/abs/2308.13451](http://arxiv.org/abs/2308.13451)

    本文提出了一种在大规模背景图中查找多个嵌入的模板图的新方法，通过迭代惩罚相似度矩阵来实现多样化匹配的发现，并提出了算法加速措施。在理论验证和实验证明中，证明了该方法的可行性和实用性。

    

    我们提出了一种在非常大的背景图中查找多个嵌入在其中的模板图的新方法。我们的方法基于Sussman等人提出的图匹配匹配滤波技术，通过在匹配滤波算法中迭代地惩罚合适的节点对相似度矩阵来实现多样化匹配的发现。此外，我们提出了算法加速，极大地提高了我们的匹配滤波方法的可扩展性。我们在相关的Erdos-Renyi图设置中对我们的方法进行了理论上的验证，显示其在温和的模型条件下能够顺序地发现多个模板。我们还通过使用模拟模型和真实世界数据集（包括人脑连接组和大型交易知识库）进行了大量实验证明了我们方法的实用性。

    We present a novel approach for finding multiple noisily embedded template graphs in a very large background graph. Our method builds upon the graph-matching-matched-filter technique proposed in Sussman et al., with the discovery of multiple diverse matchings being achieved by iteratively penalizing a suitable node-pair similarity matrix in the matched filter algorithm. In addition, we propose algorithmic speed-ups that greatly enhance the scalability of our matched-filter approach. We present theoretical justification of our methodology in the setting of correlated Erdos-Renyi graphs, showing its ability to sequentially discover multiple templates under mild model conditions. We additionally demonstrate our method's utility via extensive experiments both using simulated models and real-world dataset, include human brain connectomes and a large transactional knowledge base.
    
[^2]: Engression: 非线性回归的外推方法

    Engression: Extrapolation for Nonlinear Regression?. (arXiv:2307.00835v1 [stat.ME])

    [http://arxiv.org/abs/2307.00835](http://arxiv.org/abs/2307.00835)

    Engression是一种非线性回归方法，通过使用分布回归技术和预加性噪声模型，在训练样本范围边界外也能可靠地进行外推。

    

    外推对于许多统计学和机器学习应用至关重要，因为常常会遇到超出训练样本范围的测试数据。然而，对于非线性模型来说，外推是一个巨大的挑战。传统模型在这方面通常遇到困难：树集成模型在支持范围外提供连续的预测，而神经网络的预测往往变得不可控。这项工作旨在提供一种非线性回归方法，其可靠性在训练样本范围边界不会立即崩溃。我们的主要贡献是一种名为“engression”的新方法，它是一种预加性噪声模型的分布回归技术，其中噪声添加到协变量上并应用非线性转换。我们的实验结果表明，该模型通常适用于许多真实数据集。我们展示engression可以在一些假设下成功进行外推，例如严格限制噪声大小。

    Extrapolation is crucial in many statistical and machine learning applications, as it is common to encounter test data outside the training support. However, extrapolation is a considerable challenge for nonlinear models. Conventional models typically struggle in this regard: while tree ensembles provide a constant prediction beyond the support, neural network predictions tend to become uncontrollable. This work aims at providing a nonlinear regression methodology whose reliability does not break down immediately at the boundary of the training support. Our primary contribution is a new method called `engression' which, at its core, is a distributional regression technique for pre-additive noise models, where the noise is added to the covariates before applying a nonlinear transformation. Our experimental results indicate that this model is typically suitable for many real data sets. We show that engression can successfully perform extrapolation under some assumptions such as a strictl
    
[^3]: 广义Grenander型估计量的Bootstrap-Aided推断

    Bootstrap-Assisted Inference for Generalized Grenander-type Estimators. (arXiv:2303.13598v1 [math.ST])

    [http://arxiv.org/abs/2303.13598](http://arxiv.org/abs/2303.13598)

    本文研究了广义Grenander型估计量的大样本分布特性，提出了Bootstrap-Aided推断方法，解决了标准非参数Bootstrap难以逼近广义Grenander型估计量的大样本分布的问题。

    

    Westling和Carone（2020）提出了一个框架来研究广义Grenander型估计量的大样本分布特性，这是一类用于单调函数的非参数估计器的多才多艺的类。这些估计量的极限分布可表示为高斯过程的最大凸支撑线的左导数，该高斯过程的协方差核可以很复杂，其单项式均值可以是未知阶数（如果感兴趣的函数的平坦度未知）。标准的非参数bootstrap即使知道均值的单项式顺序，也无法一致地逼近广义Grenander型估计量的大样本分布，这使得在应用中进行统计推断成为具有挑战性的任务。为了解决这个推断问题，我们提出了一种广义Grenander型估计量的bootstrap辅助推断程序。该程序依赖于一个精心设计但自动化的变换e

    Westling and Carone (2020) proposed a framework for studying the large sample distributional properties of generalized Grenander-type estimators, a versatile class of nonparametric estimators of monotone functions. The limiting distribution of those estimators is representable as the left derivative of the greatest convex minorant of a Gaussian process whose covariance kernel can be complicated and whose monomial mean can be of unknown order (when the degree of flatness of the function of interest is unknown). The standard nonparametric bootstrap is unable to consistently approximate the large sample distribution of the generalized Grenander-type estimators even if the monomial order of the mean is known, making statistical inference a challenging endeavour in applications. To address this inferential problem, we present a bootstrap-assisted inference procedure for generalized Grenander-type estimators. The procedure relies on a carefully crafted, yet automatic, transformation of the e
    
[^4]: 基于弱相关随机过程的亚渐近极大值的高维变量聚类

    High-dimensional variable clustering based on sub-asymptotic maxima of a weakly dependent random process. (arXiv:2302.00934v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2302.00934](http://arxiv.org/abs/2302.00934)

    我们提出了一种基于亚渐近极大值的高维变量聚类模型，该模型利用群集间多变量随机过程的极大值的独立性定义种群水平的群集，我们还开发了一种无需预先指定群集数量的算法来恢复变量的群集。该算法在特定条件下能够有效地识别数据中的群集，并能够以多项式复杂度进行计算。我们的工作对于理解依赖过程的块最大值的非参数学习有重要意义，并且在神经科学领域有着应用潜力。

    

    我们提出了一种新的变量聚类模型，称为渐近独立块 (AI-block) 模型，该模型基于群集间多变量平稳混合随机过程的极大值的独立性来定义种群水平的群集。该模型类是可识别的，意味着存在一种偏序关系，允许进行统计推断。我们还提出了一种算法，无需事先指定群集的数量即可恢复变量的群集。我们的工作提供了一些理论洞察，证明了在某些条件下，我们的算法能够在计算复杂性在维度中是多项式的情况下有效地识别数据中的群集。这意味着可以非参数地学习出仅仅是亚渐近的依赖过程的块最大值的群组。为了进一步说明我们的工作的重要性，我们将我们的方法应用于神经科学数据集。

    We propose a new class of models for variable clustering called Asymptotic Independent block (AI-block) models, which defines population-level clusters based on the independence of the maxima of a multivariate stationary mixing random process among clusters. This class of models is identifiable, meaning that there exists a maximal element with a partial order between partitions, allowing for statistical inference. We also present an algorithm for recovering the clusters of variables without specifying the number of clusters \emph{a priori}. Our work provides some theoretical insights into the consistency of our algorithm, demonstrating that under certain conditions it can effectively identify clusters in the data with a computational complexity that is polynomial in the dimension. This implies that groups can be learned nonparametrically in which block maxima of a dependent process are only sub-asymptotic. To further illustrate the significance of our work, we applied our method to neu
    

