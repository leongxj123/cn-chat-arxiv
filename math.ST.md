# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Sample Complexity of Simple Binary Hypothesis Testing](https://arxiv.org/abs/2403.16981) | 该论文导出了一个公式，用于刻画简单二元假设检验的样本复杂度（乘法常数独立于$p$、$q$和所有错误参数），适用于不同的设置条件。 |

# 详细

[^1]: 简单二元假设检验的样本复杂度

    The Sample Complexity of Simple Binary Hypothesis Testing

    [https://arxiv.org/abs/2403.16981](https://arxiv.org/abs/2403.16981)

    该论文导出了一个公式，用于刻画简单二元假设检验的样本复杂度（乘法常数独立于$p$、$q$和所有错误参数），适用于不同的设置条件。

    

    简单的二元假设检验的样本复杂度是区分两个分布$p$和$q$所需的最小独立同分布样本数量，可以通过以下方式之一进行：(i) 无先验设置，类型-I错误最大为$\alpha$，类型-II错误最大为$\beta$; 或者 (ii) 贝叶斯设置，贝叶斯错误最大为$\delta$，先验分布为$(\alpha, 1-\alpha)$。 迄今为止，只在$\alpha = \beta$（无先验）或$\alpha = 1/2$（贝叶斯）时研究了此问题，并且已知样本复杂度可以用$p$和$q$之间的Hellinger散度来刻画，直到乘法常数。 在本文中，我们导出了一个公式，用来刻画样本复杂度（乘法常数独立于$p$、$q$和所有错误参数），适用于：(i) 先验设置中所有$0 \le \alpha, \beta \le 1/8$；以及 (ii) 贝叶斯设置中所有$\delta \le \alpha/4$。 特别地，该公式适用于

    arXiv:2403.16981v1 Announce Type: cross  Abstract: The sample complexity of simple binary hypothesis testing is the smallest number of i.i.d. samples required to distinguish between two distributions $p$ and $q$ in either: (i) the prior-free setting, with type-I error at most $\alpha$ and type-II error at most $\beta$; or (ii) the Bayesian setting, with Bayes error at most $\delta$ and prior distribution $(\alpha, 1-\alpha)$. This problem has only been studied when $\alpha = \beta$ (prior-free) or $\alpha = 1/2$ (Bayesian), and the sample complexity is known to be characterized by the Hellinger divergence between $p$ and $q$, up to multiplicative constants. In this paper, we derive a formula that characterizes the sample complexity (up to multiplicative constants that are independent of $p$, $q$, and all error parameters) for: (i) all $0 \le \alpha, \beta \le 1/8$ in the prior-free setting; and (ii) all $\delta \le \alpha/4$ in the Bayesian setting. In particular, the formula admits eq
    

