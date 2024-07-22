# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Global optimality under amenable symmetry constraints](https://arxiv.org/abs/2402.07613) | 该论文研究了在可接受的对称约束条件下的全局最优性问题，提出了一种满足对称性质的函数或度量，并通过引入轨道凸体和coycle等工具解决了这一问题。具体应用包括不变核均值嵌入和基于对称约束的运输方案最优性。这些结果与不变性检验的Hunt-Stein定理相关。 |
| [^2] | [Conditional Generative Models are Provably Robust: Pointwise Guarantees for Bayesian Inverse Problems.](http://arxiv.org/abs/2303.15845) | 本文证明了条件生成模型对单个观测结果有健壮性 |

# 详细

[^1]: 在可接受的对称约束条件下的全局最优性

    Global optimality under amenable symmetry constraints

    [https://arxiv.org/abs/2402.07613](https://arxiv.org/abs/2402.07613)

    该论文研究了在可接受的对称约束条件下的全局最优性问题，提出了一种满足对称性质的函数或度量，并通过引入轨道凸体和coycle等工具解决了这一问题。具体应用包括不变核均值嵌入和基于对称约束的运输方案最优性。这些结果与不变性检验的Hunt-Stein定理相关。

    

    我们研究是否存在一种满足可接受变换群指定的对称性质的函数或度量，即同时满足以下两个条件：（1）最小化给定的凸性泛函或风险，（2）满足可容忍对称约束。这种对称性质的例子包括不变性、可变性或准不变性。我们的结果依赖于Stein和Le Cam的老思想，以及在可接受群的遍历定理中出现的近似群平均值。在凸分析中，一类称为轨道凸体的凸集显得至关重要，我们在非参数设置中确定了这类轨道凸体的性质。我们还展示了一个称为coycle的简单装置如何将不同形式的对称性转化为一个问题。作为应用，我们得出了关于不变核均值嵌入和在对称约束下运输方案最优性的Monge-Kantorovich定理的结果。我们还解释了与不变性检验的Hunt-Stein定理的联系。

    We ask whether there exists a function or measure that (1) minimizes a given convex functional or risk and (2) satisfies a symmetry property specified by an amenable group of transformations. Examples of such symmetry properties are invariance, equivariance, or quasi-invariance. Our results draw on old ideas of Stein and Le Cam and on approximate group averages that appear in ergodic theorems for amenable groups. A class of convex sets known as orbitopes in convex analysis emerges as crucial, and we establish properties of such orbitopes in nonparametric settings. We also show how a simple device called a cocycle can be used to reduce different forms of symmetry to a single problem. As applications, we obtain results on invariant kernel mean embeddings and a Monge-Kantorovich theorem on optimality of transport plans under symmetry constraints. We also explain connections to the Hunt-Stein theorem on invariant tests.
    
[^2]: 条件生成模型可证明具有健壮性:银湖反问题的逐点保证

    Conditional Generative Models are Provably Robust: Pointwise Guarantees for Bayesian Inverse Problems. (arXiv:2303.15845v1 [cs.LG])

    [http://arxiv.org/abs/2303.15845](http://arxiv.org/abs/2303.15845)

    本文证明了条件生成模型对单个观测结果有健壮性

    

    条件生成模型成为采样银湖反问题后验概率的强大工具. 经典的贝叶斯文献已经知道后验测度对先前测度和负对数似然函数(包括观察的扰动)非常 robust. 但是, 就我们所知, 条件生成模型的健壮性还没被研究过. 在本文中, 我们首次证明了适当学习的条件生成模型在单个观测值方面提供了健壮的结果.

    Conditional generative models became a very powerful tool to sample from Bayesian inverse problem posteriors. It is well-known in classical Bayesian literature that posterior measures are quite robust with respect to perturbations of both the prior measure and the negative log-likelihood, which includes perturbations of the observations. However, to the best of our knowledge, the robustness of conditional generative models with respect to perturbations of the observations has not been investigated yet. In this paper, we prove for the first time that appropriately learned conditional generative models provide robust results for single observations.
    

