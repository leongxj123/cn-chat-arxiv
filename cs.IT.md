# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Theoretical guarantees on the best-of-n alignment policy.](http://arxiv.org/abs/2401.01879) | 该论文研究了对齐生成模型的最佳n对齐策略，并证明了之前文献中的某个分析表达式是错误的。研究者们提出了一个新的KL散度估计方法，并通过实验证明其有效性。 |

# 详细

[^1]: 关于最佳n对齐策略的理论保证

    Theoretical guarantees on the best-of-n alignment policy. (arXiv:2401.01879v1 [cs.LG])

    [http://arxiv.org/abs/2401.01879](http://arxiv.org/abs/2401.01879)

    该论文研究了对齐生成模型的最佳n对齐策略，并证明了之前文献中的某个分析表达式是错误的。研究者们提出了一个新的KL散度估计方法，并通过实验证明其有效性。

    

    一个简单有效的生成模型对齐方法是最佳n对齐策略，该策略从一个基本策略中抽取n个样本，并根据奖励函数对它们进行排序，选择排名最高的样本。文献中常用的分析表达式声称最佳n对齐策略与基本策略之间的KL散度等于$\log (n) (n-1)/n$。我们证明了该论断的不正确性，并展示了它只是实际KL散度的一个上界。我们还研究了在不同情况下该上界的紧致性。最后，我们提出了一种新的KL散度估计方法，并通过几个例子的实验证明它能提供一个紧致的近似。

    A simple and effective method for the alignment of generative models is the best-of-$n$ policy, where $n$ samples are drawn from a base policy, and ranked based on a reward function, and the highest ranking one is selected. A commonly used analytical expression in the literature claims that the KL divergence between the best-of-$n$ policy and the base policy is equal to $\log (n) (n-1)/n.$ We disprove the validity of this claim, and show that it is an upper bound on the actual KL divergence. We also explore the tightness of this upper bound in different regimes. Finally, we propose a new estimator for the KL divergence and empirically show that it provides a tight approximation through a few examples.
    

