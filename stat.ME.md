# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Uncertainty estimation in spatial interpolation of satellite precipitation with ensemble learning](https://arxiv.org/abs/2403.10567) | 引入九种集成学习器并利用新颖特征工程策略，结合多种分位数回归算法，填补了空间插值中集成学习的不确定性估计领域的研究空白 |
| [^2] | [Yurinskii's Coupling for Martingales](https://arxiv.org/abs/2210.00362) | Yurinskii的耦合方法在$\ell^p$-范数下提供了更弱条件下的逼近马丁格尔，同时引入了更一般的高斯混合分布，并提供了第三阶耦合方法以在某些情况下获得更紧密的逼近。 |

# 详细

[^1]: 集成学习中的卫星降水空间插值不确定性估计

    Uncertainty estimation in spatial interpolation of satellite precipitation with ensemble learning

    [https://arxiv.org/abs/2403.10567](https://arxiv.org/abs/2403.10567)

    引入九种集成学习器并利用新颖特征工程策略，结合多种分位数回归算法，填补了空间插值中集成学习的不确定性估计领域的研究空白

    

    arXiv:2403.10567v1 公告类型：新的 摘要：概率分布形式的预测对决策至关重要。分位数回归在空间插值设置中能够合并遥感和雨量数据，实现此目标。然而，在这种情境下，分位数回归算法的集成学习尚未被研究。本文通过引入九种基于分位数的集成学习器并将其应用于大型降水数据集来填补这一空白。我们采用了一种新颖的特征工程策略，将预测因子减少为相关位置的加权距离卫星降水，结合位置高程。我们的集成学习器包括六种堆叠方法和三种简单方法（均值、中位数、最佳组合器），结合了六种个体算法：分位数回归(QR)、分位数回归森林(QRF)、广义随机森林(GRF)、梯度提升机(GBM)、轻量级梯度提升机(LightGBM)和分位数回归神经网络

    arXiv:2403.10567v1 Announce Type: new  Abstract: Predictions in the form of probability distributions are crucial for decision-making. Quantile regression enables this within spatial interpolation settings for merging remote sensing and gauge precipitation data. However, ensemble learning of quantile regression algorithms remains unexplored in this context. Here, we address this gap by introducing nine quantile-based ensemble learners and applying them to large precipitation datasets. We employed a novel feature engineering strategy, reducing predictors to distance-weighted satellite precipitation at relevant locations, combined with location elevation. Our ensemble learners include six stacking and three simple methods (mean, median, best combiner), combining six individual algorithms: quantile regression (QR), quantile regression forests (QRF), generalized random forests (GRF), gradient boosting machines (GBM), light gradient boosting machines (LightGBM), and quantile regression neur
    
[^2]: Yurinskii的马丁格尔耦合

    Yurinskii's Coupling for Martingales

    [https://arxiv.org/abs/2210.00362](https://arxiv.org/abs/2210.00362)

    Yurinskii的耦合方法在$\ell^p$-范数下提供了更弱条件下的逼近马丁格尔，同时引入了更一般的高斯混合分布，并提供了第三阶耦合方法以在某些情况下获得更紧密的逼近。

    

    Yurinskii的耦合是数学统计和应用概率中一种常用的非渐近分布分析理论工具，提供了在易于验证条件下具有显式误差界限的高斯强逼近。最初在独立随机向量和为的$\ell^2$-范数中陈述，最近已将其扩展到$1 \leq p \leq \infty$时的$\ell^p$-范数，以及在某些强条件下的$\ell^2$-范数的向量值鞅。我们的主要结果是在远比之前施加的条件更弱的情况下，在$\ell^p$-范数下提供了逼近马丁格尔的Yurinskii耦合。我们的公式进一步允许耦合变量遵循更一般的高斯混合分布，并且我们提供了一种新颖的第三阶耦合方法，在某些情况下提供更紧密的逼近。我们将我们的主要结果专门应用于混合马丁格尔，马丁格尔和其他情况。

    arXiv:2210.00362v2 Announce Type: replace-cross  Abstract: Yurinskii's coupling is a popular theoretical tool for non-asymptotic distributional analysis in mathematical statistics and applied probability, offering a Gaussian strong approximation with an explicit error bound under easily verified conditions. Originally stated in $\ell^2$-norm for sums of independent random vectors, it has recently been extended both to the $\ell^p$-norm, for $1 \leq p \leq \infty$, and to vector-valued martingales in $\ell^2$-norm, under some strong conditions. We present as our main result a Yurinskii coupling for approximate martingales in $\ell^p$-norm, under substantially weaker conditions than those previously imposed. Our formulation further allows for the coupling variable to follow a more general Gaussian mixture distribution, and we provide a novel third-order coupling method which gives tighter approximations in certain settings. We specialize our main result to mixingales, martingales, and in
    

