# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bayesian Quantile Regression with Subset Selection: A Posterior Summarization Perspective.](http://arxiv.org/abs/2311.02043) | 本研究提出了一种基于贝叶斯决策分析的方法，对于任何贝叶斯回归模型，可以得到每个条件分位数的最佳和可解释的线性估计值和不确定性量化。该方法是一种适用于特定分位数子集选择的有效工具。 |
| [^2] | [Understanding black-box models with dependent inputs through a generalization of Hoeffding's decomposition.](http://arxiv.org/abs/2310.06567) | 通过提出一个新的框架，我们可以解释有关依赖输入的黑箱模型。我们证明了在一些合理的假设下，非线性函数可以唯一分解为每个可能子集的函数之和。这个框架有效地推广了Hoeffding分解，并提供了新颖的可解释性指标。 |

# 详细

[^1]: 基于子集选择的贝叶斯分位回归：后验总结视角

    Bayesian Quantile Regression with Subset Selection: A Posterior Summarization Perspective. (arXiv:2311.02043v1 [stat.ME])

    [http://arxiv.org/abs/2311.02043](http://arxiv.org/abs/2311.02043)

    本研究提出了一种基于贝叶斯决策分析的方法，对于任何贝叶斯回归模型，可以得到每个条件分位数的最佳和可解释的线性估计值和不确定性量化。该方法是一种适用于特定分位数子集选择的有效工具。

    

    分位回归是一种强大的工具，用于推断协变量如何影响响应分布的特定分位数。现有方法要么分别估计每个感兴趣分位数的条件分位数，要么使用半参数或非参数模型估计整个条件分布。前者经常产生不适合实际数据的模型，并且不在分位数之间共享信息，而后者则以复杂且受限制的模型为特点，难以解释和计算效率低下。此外，这两种方法都不适合于特定分位数的子集选择。相反，我们从贝叶斯决策分析的角度出发，提出了线性分位估计、不确定性量化和子集选择的基本问题。对于任何贝叶斯回归模型，我们为每个基于模型的条件分位数推导出最佳和可解释的线性估计值和不确定性量化。我们的方法引入了一种分位数聚焦的方法。

    Quantile regression is a powerful tool for inferring how covariates affect specific percentiles of the response distribution. Existing methods either estimate conditional quantiles separately for each quantile of interest or estimate the entire conditional distribution using semi- or non-parametric models. The former often produce inadequate models for real data and do not share information across quantiles, while the latter are characterized by complex and constrained models that can be difficult to interpret and computationally inefficient. Further, neither approach is well-suited for quantile-specific subset selection. Instead, we pose the fundamental problems of linear quantile estimation, uncertainty quantification, and subset selection from a Bayesian decision analysis perspective. For any Bayesian regression model, we derive optimal and interpretable linear estimates and uncertainty quantification for each model-based conditional quantile. Our approach introduces a quantile-focu
    
[^2]: 通过Hoeffding分解的推广，理解有关依赖输入的黑箱模型

    Understanding black-box models with dependent inputs through a generalization of Hoeffding's decomposition. (arXiv:2310.06567v1 [math.FA])

    [http://arxiv.org/abs/2310.06567](http://arxiv.org/abs/2310.06567)

    通过提出一个新的框架，我们可以解释有关依赖输入的黑箱模型。我们证明了在一些合理的假设下，非线性函数可以唯一分解为每个可能子集的函数之和。这个框架有效地推广了Hoeffding分解，并提供了新颖的可解释性指标。

    

    解释黑箱模型的主要挑战之一是能够将非互不相关随机输入的平方可积函数唯一分解为每个可能子集的函数之和。然而，处理输入之间的依赖关系可能很复杂。我们提出了一个新的框架来研究这个问题，将三个数学领域联系起来：概率论、函数分析和组合数学。我们表明，在输入上的两个合理假设下（非完美的函数依赖性和非退化的随机依赖性），总是可以唯一分解这样一个函数。这种“规范分解”相对直观，揭示了非线性相关输入的非线性函数的线性特性。在这个框架中，我们有效地推广了众所周知的Hoeffding分解，可以看作是一个特殊情况。黑箱模型的斜投影为新颖的可解释性指标提供了可能。

    One of the main challenges for interpreting black-box models is the ability to uniquely decompose square-integrable functions of non-mutually independent random inputs into a sum of functions of every possible subset of variables. However, dealing with dependencies among inputs can be complicated. We propose a novel framework to study this problem, linking three domains of mathematics: probability theory, functional analysis, and combinatorics. We show that, under two reasonable assumptions on the inputs (non-perfect functional dependence and non-degenerate stochastic dependence), it is always possible to decompose uniquely such a function. This ``canonical decomposition'' is relatively intuitive and unveils the linear nature of non-linear functions of non-linearly dependent inputs. In this framework, we effectively generalize the well-known Hoeffding decomposition, which can be seen as a particular case. Oblique projections of the black-box model allow for novel interpretability indic
    

