# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Computational-Statistical Gaps for Improper Learning in Sparse Linear Regression](https://arxiv.org/abs/2402.14103) | 该研究探讨了稀疏线性回归中的计算统计差距问题，为了高效地找到可以在样本上实现非平凡预测误差的潜在密集估计的回归向量，需要至少 $\Omega(k \log (d/k))$ 个样本。 |

# 详细

[^1]: 稀疏线性回归中不当学习的计算统计差距

    Computational-Statistical Gaps for Improper Learning in Sparse Linear Regression

    [https://arxiv.org/abs/2402.14103](https://arxiv.org/abs/2402.14103)

    该研究探讨了稀疏线性回归中的计算统计差距问题，为了高效地找到可以在样本上实现非平凡预测误差的潜在密集估计的回归向量，需要至少 $\Omega(k \log (d/k))$ 个样本。

    

    我们研究了稀疏线性回归中不当学习的计算统计差距。具体来说，给定来自维度为 $d$ 的 $k$-稀疏线性模型的 $n$ 个样本，我们询问了在时间多项式中的最小样本复杂度，以便高效地找到一个对这 $n$ 个样本达到非平凡预测误差的潜在密集估计的回归向量。信息理论上，这可以用 $\Theta(k \log (d/k))$ 个样本实现。然而，尽管在文献中很显著，但没有已知的多项式时间算法可以在不附加对模型的其他限制的情况下使用少于 $\Theta(d)$ 个样本达到相同的保证。类似地，现有的困难结果要么仅限于适当设置，在该设置中估计值也必须是稀疏的，要么仅适用于特定算法。

    arXiv:2402.14103v1 Announce Type: new  Abstract: We study computational-statistical gaps for improper learning in sparse linear regression. More specifically, given $n$ samples from a $k$-sparse linear model in dimension $d$, we ask what is the minimum sample complexity to efficiently (in time polynomial in $d$, $k$, and $n$) find a potentially dense estimate for the regression vector that achieves non-trivial prediction error on the $n$ samples. Information-theoretically this can be achieved using $\Theta(k \log (d/k))$ samples. Yet, despite its prominence in the literature, there is no polynomial-time algorithm known to achieve the same guarantees using less than $\Theta(d)$ samples without additional restrictions on the model. Similarly, existing hardness results are either restricted to the proper setting, in which the estimate must be sparse as well, or only apply to specific algorithms.   We give evidence that efficient algorithms for this task require at least (roughly) $\Omega(
    

