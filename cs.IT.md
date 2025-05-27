# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Sample Complexity of Simple Binary Hypothesis Testing](https://arxiv.org/abs/2403.16981) | 该论文导出了一个公式，用于刻画简单二元假设检验的样本复杂度（乘法常数独立于$p$、$q$和所有错误参数），适用于不同的设置条件。 |
| [^2] | [Joint Communication and Computation Framework for Goal-Oriented Semantic Communication with Distortion Rate Resilience.](http://arxiv.org/abs/2309.14587) | 本论文提出了一个创新的联合通信和计算框架，利用率畸变理论来分析通信和语义压缩引起的畸变，从而评估其对目标导向语义通信中人工智能模型性能的影响，使目标导向语义通信问题成为可能。 |

# 详细

[^1]: 简单二元假设检验的样本复杂度

    The Sample Complexity of Simple Binary Hypothesis Testing

    [https://arxiv.org/abs/2403.16981](https://arxiv.org/abs/2403.16981)

    该论文导出了一个公式，用于刻画简单二元假设检验的样本复杂度（乘法常数独立于$p$、$q$和所有错误参数），适用于不同的设置条件。

    

    简单的二元假设检验的样本复杂度是区分两个分布$p$和$q$所需的最小独立同分布样本数量，可以通过以下方式之一进行：(i) 无先验设置，类型-I错误最大为$\alpha$，类型-II错误最大为$\beta$; 或者 (ii) 贝叶斯设置，贝叶斯错误最大为$\delta$，先验分布为$(\alpha, 1-\alpha)$。 迄今为止，只在$\alpha = \beta$（无先验）或$\alpha = 1/2$（贝叶斯）时研究了此问题，并且已知样本复杂度可以用$p$和$q$之间的Hellinger散度来刻画，直到乘法常数。 在本文中，我们导出了一个公式，用来刻画样本复杂度（乘法常数独立于$p$、$q$和所有错误参数），适用于：(i) 先验设置中所有$0 \le \alpha, \beta \le 1/8$；以及 (ii) 贝叶斯设置中所有$\delta \le \alpha/4$。 特别地，该公式适用于

    arXiv:2403.16981v1 Announce Type: cross  Abstract: The sample complexity of simple binary hypothesis testing is the smallest number of i.i.d. samples required to distinguish between two distributions $p$ and $q$ in either: (i) the prior-free setting, with type-I error at most $\alpha$ and type-II error at most $\beta$; or (ii) the Bayesian setting, with Bayes error at most $\delta$ and prior distribution $(\alpha, 1-\alpha)$. This problem has only been studied when $\alpha = \beta$ (prior-free) or $\alpha = 1/2$ (Bayesian), and the sample complexity is known to be characterized by the Hellinger divergence between $p$ and $q$, up to multiplicative constants. In this paper, we derive a formula that characterizes the sample complexity (up to multiplicative constants that are independent of $p$, $q$, and all error parameters) for: (i) all $0 \le \alpha, \beta \le 1/8$ in the prior-free setting; and (ii) all $\delta \le \alpha/4$ in the Bayesian setting. In particular, the formula admits eq
    
[^2]: 目标导向语义通信的联合通信和计算框架，具有畸变率鲁棒性

    Joint Communication and Computation Framework for Goal-Oriented Semantic Communication with Distortion Rate Resilience. (arXiv:2309.14587v1 [cs.LG])

    [http://arxiv.org/abs/2309.14587](http://arxiv.org/abs/2309.14587)

    本论文提出了一个创新的联合通信和计算框架，利用率畸变理论来分析通信和语义压缩引起的畸变，从而评估其对目标导向语义通信中人工智能模型性能的影响，使目标导向语义通信问题成为可能。

    

    最近关于语义通信的研究主要考虑准确性作为优化目标导向通信系统的主要问题。然而，这些方法引入了一个悖论：人工智能任务的准确性应该通过训练自然地出现，而不是由网络约束所决定。鉴于这个困境，本文引入了一种创新的方法，利用率畸变理论来分析由通信和语义压缩引起的畸变，并分析学习过程。具体来说，我们研究了原始数据和畸变数据之间的分布偏移，从而评估其对人工智能模型性能的影响。基于这个分析，我们可以预先估计人工智能任务的实际准确性，使目标导向语义通信问题变得可行。为了实现这个目标，我们提出了我们方法的理论基础，并进行了模拟和实验。

    Recent research efforts on semantic communication have mostly considered accuracy as a main problem for optimizing goal-oriented communication systems. However, these approaches introduce a paradox: the accuracy of artificial intelligence (AI) tasks should naturally emerge through training rather than being dictated by network constraints. Acknowledging this dilemma, this work introduces an innovative approach that leverages the rate-distortion theory to analyze distortions induced by communication and semantic compression, thereby analyzing the learning process. Specifically, we examine the distribution shift between the original data and the distorted data, thus assessing its impact on the AI model's performance. Founding upon this analysis, we can preemptively estimate the empirical accuracy of AI tasks, making the goal-oriented semantic communication problem feasible. To achieve this objective, we present the theoretical foundation of our approach, accompanied by simulations and ex
    

