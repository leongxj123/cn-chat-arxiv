# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Minimax-Regret Sample Selection in Randomized Experiments](https://arxiv.org/abs/2403.01386) | 通过最小后悔框架，提出了在随机试验中优化样本选择以实现异质人群中最佳福利的问题，并在不同条件下推导出最优的样本选择方案。 |
| [^2] | [Accelerating Look-ahead in Bayesian Optimization: Multilevel Monte Carlo is All you Need](https://arxiv.org/abs/2402.02111) | 本文利用多层蒙特卡洛方法加速贝叶斯优化中的前瞻过程，并证明在涉及嵌套期望和最大化的问题中具有优势。 |
| [^3] | [An Embedded Diachronic Sense Change Model with a Case Study from Ancient Greek.](http://arxiv.org/abs/2311.00541) | 本论文介绍了一个嵌入式历时语义变化模型（EDiSC），结合了词嵌入和DiSC模型，通过无监督学习分析古希腊文本中目标词汇的意义变化。实验证明EDiSC具有优越的性能。 |

# 详细

[^1]: 随机实验中的最小后悔样本选择

    Minimax-Regret Sample Selection in Randomized Experiments

    [https://arxiv.org/abs/2403.01386](https://arxiv.org/abs/2403.01386)

    通过最小后悔框架，提出了在随机试验中优化样本选择以实现异质人群中最佳福利的问题，并在不同条件下推导出最优的样本选择方案。

    

    随机对照试验（RCTs）经常在存在许多可能对所评估的治疗效果有差异的子人群中进行。我们考虑了样本选择问题，即在异质人群中如何选择入组RRT，以优化福利。我们在最小后悔框架下形式化了这个问题，并在多种条件下推导出最优的样本选择方案。我们还强调了不同的目标和决策如何导致明显不同的关于最佳样本分配的指导，通过利用历史COVID-19试验数据进行了一项合成实验。

    arXiv:2403.01386v1 Announce Type: cross  Abstract: Randomized controlled trials (RCTs) are often run in settings with many subpopulations that may have differential benefits from the treatment being evaluated. We consider the problem of sample selection, i.e., whom to enroll in an RCT, such as to optimize welfare in a heterogeneous population. We formalize this problem within the minimax-regret framework, and derive optimal sample-selection schemes under a variety of conditions. We also highlight how different objectives and decisions can lead to notably different guidance regarding optimal sample allocation through a synthetic experiment leveraging historical COVID-19 trial data.
    
[^2]: 加速贝叶斯优化中的前瞻：多层蒙特卡洛就够了

    Accelerating Look-ahead in Bayesian Optimization: Multilevel Monte Carlo is All you Need

    [https://arxiv.org/abs/2402.02111](https://arxiv.org/abs/2402.02111)

    本文利用多层蒙特卡洛方法加速贝叶斯优化中的前瞻过程，并证明在涉及嵌套期望和最大化的问题中具有优势。

    

    我们利用多层蒙特卡洛(MLMC)来提高涉及嵌套期望和最大化的多步前瞻贝叶斯优化(BO)方法的性能。普通蒙特卡洛的复杂度在嵌套操作中会降低，而MLMC能够以规范蒙特卡洛收敛速度解决这类问题，而且不依赖于维度和平滑性假设。我们的理论研究主要关注一步和两步前瞻采集函数的近似改进，但正如我们所讨论的，这种方法在多种方面是可推广的，包括超越BO的背景。我们通过数值验证了我们的发现，并在几个基准示例中展示了MLMC在BO中的优势。代码在这里获取：https://github.com/Shangda-Yang/MLMCBO。

    We leverage multilevel Monte Carlo (MLMC) to improve the performance of multi-step look-ahead Bayesian optimization (BO) methods that involve nested expectations and maximizations. The complexity rate of naive Monte Carlo degrades for nested operations, whereas MLMC is capable of achieving the canonical Monte Carlo convergence rate for this type of problem, independently of dimension and without any smoothness assumptions. Our theoretical study focuses on the approximation improvements for one- and two-step look-ahead acquisition functions, but, as we discuss, the approach is generalizable in various ways, including beyond the context of BO. Findings are verified numerically and the benefits of MLMC for BO are illustrated on several benchmark examples. Code is available here https://github.com/Shangda-Yang/MLMCBO.
    
[^3]: 一个带有嵌入式历时语义变化模型的论文与一个关于古希腊的案例研究

    An Embedded Diachronic Sense Change Model with a Case Study from Ancient Greek. (arXiv:2311.00541v1 [cs.CL])

    [http://arxiv.org/abs/2311.00541](http://arxiv.org/abs/2311.00541)

    本论文介绍了一个嵌入式历时语义变化模型（EDiSC），结合了词嵌入和DiSC模型，通过无监督学习分析古希腊文本中目标词汇的意义变化。实验证明EDiSC具有优越的性能。

    

    词汇的意义随着时间的推移而变化，词义在这个过程中会演变、出现或消失。对于古代语言来说，由于语料库通常较小、稀疏且嘈杂，准确建模这种变化变得具有挑战性，因此对于意义变化估计的不确定性进行量化变得重要。GASC和DiSC是现有的生成模型，已经被用来分析古希腊文本语料库中目标词汇的意义变化，使用了无监督学习并没有借助任何预训练的帮助。这些模型将给定目标词汇（如"kosmos"，意为装饰、秩序或世界）的意义表示为上下文词汇的分布，并将意义的普遍性表示为意义的分布。这些模型使用马尔科夫链蒙特卡洛方法进行拟合，以测量这些表示中的时间变化。在本文中，我们介绍了EDiSC，这是DiSC的嵌入版本，它将词嵌入与DiSC相结合，提供了更优秀的模型性能。我们通过实验证明，EDiSC提供了改进的性能。

    Word meanings change over time, and word senses evolve, emerge or die out in the process. For ancient languages, where the corpora are often small, sparse and noisy, modelling such changes accurately proves challenging, and quantifying uncertainty in sense-change estimates consequently becomes important. GASC and DiSC are existing generative models that have been used to analyse sense change for target words from an ancient Greek text corpus, using unsupervised learning without the help of any pre-training. These models represent the senses of a given target word such as "kosmos" (meaning decoration, order or world) as distributions over context words, and sense prevalence as a distribution over senses. The models are fitted using MCMC methods to measure temporal changes in these representations. In this paper, we introduce EDiSC, an embedded version of DiSC, which combines word embeddings with DiSC to provide superior model performance. We show empirically that EDiSC offers improved p
    

