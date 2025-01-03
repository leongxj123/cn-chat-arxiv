# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Online Joint Assortment-Inventory Optimization under MNL Choices.](http://arxiv.org/abs/2304.02022) | 本文提出了一个算法解决在线联合组合库存优化问题，能够在平衡探索与开发的措施下实现最大化预期总利润，并为该算法建立了遗憾上界。 |

# 详细

[^1]: 基于MNL选择模型的在线联合组合库存优化问题研究

    Online Joint Assortment-Inventory Optimization under MNL Choices. (arXiv:2304.02022v1 [cs.LG])

    [http://arxiv.org/abs/2304.02022](http://arxiv.org/abs/2304.02022)

    本文提出了一个算法解决在线联合组合库存优化问题，能够在平衡探索与开发的措施下实现最大化预期总利润，并为该算法建立了遗憾上界。

    

    本文研究了一种在线联合组合库存优化问题，在该问题中，我们假设每个顾客的选择行为都遵循Multinomial Logit（MNL）选择模型，吸引力参数是先验未知的。零售商进行周期性组合和库存决策，以动态地从实现的需求中学习吸引力参数，同时在时间上最大化预期的总利润。本文提出了一种新算法，可以有效地平衡组合和库存在线决策中的探索和开发。我们的算法建立在一个新的MNL吸引力参数估计器，一种通过自适应调整某些已知和未知参数来激励探索的新方法，以及一个用于静态单周期组合库存规划问题的优化oracle基础之上。我们为我们的算法建立了遗憾上界，以及关于在线联合组合库存优化问题的下界。

    We study an online joint assortment-inventory optimization problem, in which we assume that the choice behavior of each customer follows the Multinomial Logit (MNL) choice model, and the attraction parameters are unknown a priori. The retailer makes periodic assortment and inventory decisions to dynamically learn from the realized demands about the attraction parameters while maximizing the expected total profit over time. In this paper, we propose a novel algorithm that can effectively balance the exploration and exploitation in the online decision-making of assortment and inventory. Our algorithm builds on a new estimator for the MNL attraction parameters, a novel approach to incentivize exploration by adaptively tuning certain known and unknown parameters, and an optimization oracle to static single-cycle assortment-inventory planning problems with given parameters. We establish a regret upper bound for our algorithm and a lower bound for the online joint assortment-inventory optimi
    

