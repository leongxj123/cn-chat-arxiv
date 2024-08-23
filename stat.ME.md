# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Uncertainty estimation in satellite precipitation interpolation with machine learning](https://arxiv.org/abs/2311.07511) | 该研究使用机器学习算法对卫星和测站数据进行插值，通过量化预测不确定性来提高降水数据集的分辨率。 |
| [^2] | [Estimating Treatment Effects using Multiple Surrogates: The Role of the Surrogate Score and the Surrogate Index](https://arxiv.org/abs/1603.09326) | 利用现代数据集中大量中间结果的事实，即使没有单个替代指标满足统计替代条件，使用多个替代指标也可能是有效的。 |

# 详细

[^1]: 用机器学习进行卫星降水插值的不确定性估计

    Uncertainty estimation in satellite precipitation interpolation with machine learning

    [https://arxiv.org/abs/2311.07511](https://arxiv.org/abs/2311.07511)

    该研究使用机器学习算法对卫星和测站数据进行插值，通过量化预测不确定性来提高降水数据集的分辨率。

    

    合并卫星和测站数据并利用机器学习产生高分辨率降水数据集，但预测不确定性估计往往缺失。我们通过对比六种算法，大部分是针对这一任务而设计的新算法，来量化空间插值中的预测不确定性。在连续美国的15年月度数据上，我们比较了分位数回归（QR）、分位数回归森林（QRF）、广义随机森林（GRF）、梯度提升机（GBM）、轻梯度提升机（LightGBM）和分位数回归神经网络（QRNN）。它们能够在九个分位水平（0.025、0.050、0.100、0.250、0.500、0.750、0.900、0.950、0.975）上发布预测降水分位数，以近似完整概率分布，评估时采用分位数评分函数和分位数评分规则。特征重要性分析揭示了卫星降水（PERSIA

    arXiv:2311.07511v2 Announce Type: replace-cross  Abstract: Merging satellite and gauge data with machine learning produces high-resolution precipitation datasets, but uncertainty estimates are often missing. We address this gap by benchmarking six algorithms, mostly novel for this task, for quantifying predictive uncertainty in spatial interpolation. On 15 years of monthly data over the contiguous United States (CONUS), we compared quantile regression (QR), quantile regression forests (QRF), generalized random forests (GRF), gradient boosting machines (GBM), light gradient boosting machines (LightGBM), and quantile regression neural networks (QRNN). Their ability to issue predictive precipitation quantiles at nine quantile levels (0.025, 0.050, 0.100, 0.250, 0.500, 0.750, 0.900, 0.950, 0.975), approximating the full probability distribution, was evaluated using quantile scoring functions and the quantile scoring rule. Feature importance analysis revealed satellite precipitation (PERSIA
    
[^2]: 利用多个替代指标估计治疗效果：替代分数和替代指数的作用

    Estimating Treatment Effects using Multiple Surrogates: The Role of the Surrogate Score and the Surrogate Index

    [https://arxiv.org/abs/1603.09326](https://arxiv.org/abs/1603.09326)

    利用现代数据集中大量中间结果的事实，即使没有单个替代指标满足统计替代条件，使用多个替代指标也可能是有效的。

    

    估计治疗效果长期作用是许多领域感兴趣的问题。 估计此类治疗效果的一个常见挑战在于长期结果在需要做出政策决策的时间范围内是未观察到的。 克服这种缺失数据问题的一种方法是分析治疗效果对中间结果的影响，通常称为统计替代指标，如果满足条件：在统计替代指标的条件下，治疗和结果是独立的。  替代条件的有效性经常是有争议的。 在现代数据集中，研究人员通常观察到大量中间结果，可能是数百个或数千个，被认为位于治疗和长期感兴趣的结果之间的因果链上或附近。 即使没有个别代理满足统计替代条件，使用多个代理也可以。

    arXiv:1603.09326v4 Announce Type: replace-cross  Abstract: Estimating the long-term effects of treatments is of interest in many fields. A common challenge in estimating such treatment effects is that long-term outcomes are unobserved in the time frame needed to make policy decisions. One approach to overcome this missing data problem is to analyze treatments effects on an intermediate outcome, often called a statistical surrogate, if it satisfies the condition that treatment and outcome are independent conditional on the statistical surrogate. The validity of the surrogacy condition is often controversial. Here we exploit that fact that in modern datasets, researchers often observe a large number, possibly hundreds or thousands, of intermediate outcomes, thought to lie on or close to the causal chain between the treatment and the long-term outcome of interest. Even if none of the individual proxies satisfies the statistical surrogacy criterion by itself, using multiple proxies can be 
    

