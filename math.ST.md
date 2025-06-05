# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Batched Nonparametric Contextual Bandits](https://arxiv.org/abs/2402.17732) | 该论文研究了批处理约束下的非参数上下文臂问题，提出了一种名为BaSEDB的方案，在动态分割协变量空间的同时，实现了最优的后悔。 |

# 详细

[^1]: 批处理非参数上下文臂

    Batched Nonparametric Contextual Bandits

    [https://arxiv.org/abs/2402.17732](https://arxiv.org/abs/2402.17732)

    该论文研究了批处理约束下的非参数上下文臂问题，提出了一种名为BaSEDB的方案，在动态分割协变量空间的同时，实现了最优的后悔。

    

    我们研究了在批处理约束下的非参数上下文臂问题，在这种情况下，每个动作的期望奖励被建模为协变量的平滑函数，并且策略更新是在每个Observations批次结束时进行的。我们为这种设置建立了一个最小化后悔的下限，并提出了一种名为Batched Successive Elimination with Dynamic Binning（BaSEDB）的方案，可以实现最优的后悔（达到对数因子）。实质上，BaSEDB动态地将协变量空间分割成更小的箱子，并仔细调整它们的宽度以符合批次大小。我们还展示了在批处理约束下静态分箱的非最优性，突出了动态分箱的必要性。另外，我们的结果表明，在完全在线设置中，几乎恒定数量的策略更新可以达到最佳后悔。

    arXiv:2402.17732v1 Announce Type: cross  Abstract: We study nonparametric contextual bandits under batch constraints, where the expected reward for each action is modeled as a smooth function of covariates, and the policy updates are made at the end of each batch of observations. We establish a minimax regret lower bound for this setting and propose Batched Successive Elimination with Dynamic Binning (BaSEDB) that achieves optimal regret (up to logarithmic factors). In essence, BaSEDB dynamically splits the covariate space into smaller bins, carefully aligning their widths with the batch size. We also show the suboptimality of static binning under batch constraints, highlighting the necessity of dynamic binning. Additionally, our results suggest that a nearly constant number of policy updates can attain optimal regret in the fully online setting.
    

