# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Resilience of the quadratic Littlewood-Offord problem](https://arxiv.org/abs/2402.10504) | 论文研究了二次Littlewood-Offord问题的统计鲁棒性，估计了对抗性噪声对二次Radamecher混沌的影响，并提供了对二次和双线性Rademacher混沌的统计鲁棒性的下限估计。 |
| [^2] | [The Best Time for an Update: Risk-Sensitive Minimization of Age-Based Metrics.](http://arxiv.org/abs/2401.10265) | 该论文研究了基于风险敏感的年龄度量最小化问题，提出了一种新的风险状态概念和风险度量方法，并介绍了两种风险敏感策略。 |
| [^3] | [Nystr\"om $M$-Hilbert-Schmidt Independence Criterion.](http://arxiv.org/abs/2302.09930) | 这项研究提出了Nystr\"om $M$-Hilbert-Schmidt独立准则，针对大规模应用的二次计算瓶颈问题进行了解决，并兼顾了多个随机变量的推广情况和理论保证。 |

# 详细

[^1]: 二次Littlewood-Offord问题的弹性

    Resilience of the quadratic Littlewood-Offord problem

    [https://arxiv.org/abs/2402.10504](https://arxiv.org/abs/2402.10504)

    论文研究了二次Littlewood-Offord问题的统计鲁棒性，估计了对抗性噪声对二次Radamecher混沌的影响，并提供了对二次和双线性Rademacher混沌的统计鲁棒性的下限估计。

    

    我们研究了高维数据的统计鲁棒性。我们的结果提供了关于对抗性噪声对二次Radamecher混沌$\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi}$反集中特性的影响的估计，其中$M$是一个固定的（高维）矩阵，$\boldsymbol{\xi}$是一个共形Rademacher向量。具体来说，我们探讨了$\boldsymbol{\xi}$能够承受多少对抗性符号翻转而不“膨胀”$\sup_{x\in \mathbb{R}} \mathbb{P} \left\{\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi} = x\right\}$，从而“去除”原始分布导致更“有粒度”和对抗性偏倚的分布。我们的结果为二次和双线性Rademacher混沌的统计鲁棒性提供了下限估计；这些结果在关键区域被证明是渐近紧的。

    arXiv:2402.10504v1 Announce Type: cross  Abstract: We study the statistical resilience of high-dimensional data. Our results provide estimates as to the effects of adversarial noise over the anti-concentration properties of the quadratic Radamecher chaos $\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi}$, where $M$ is a fixed (high-dimensional) matrix and $\boldsymbol{\xi}$ is a conformal Rademacher vector. Specifically, we pursue the question of how many adversarial sign-flips can $\boldsymbol{\xi}$ sustain without "inflating" $\sup_{x\in \mathbb{R}} \mathbb{P} \left\{\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi} = x\right\}$ and thus "de-smooth" the original distribution resulting in a more "grainy" and adversarially biased distribution. Our results provide lower bound estimations for the statistical resilience of the quadratic and bilinear Rademacher chaos; these are shown to be asymptotically tight across key regimes.
    
[^2]: 最佳更新时间：基于风险敏感的年龄度量最小化

    The Best Time for an Update: Risk-Sensitive Minimization of Age-Based Metrics. (arXiv:2401.10265v1 [cs.IT])

    [http://arxiv.org/abs/2401.10265](http://arxiv.org/abs/2401.10265)

    该论文研究了基于风险敏感的年龄度量最小化问题，提出了一种新的风险状态概念和风险度量方法，并介绍了两种风险敏感策略。

    

    流行的量化传输数据质量的方法包括信息年龄（Age of Information，AoI），查询信息年龄（Query Age of Information，QAoI）和不正确信息年龄（Age of Incorrect Information，AoII）。我们考虑在点对点无线通信系统中使用这些度量，发送方监视一个进程并向接收方发送状态更新。我们的挑战是决定更新的最佳时间，平衡传输能量和接收方的年龄度量。由于高年龄度量值引起的不稳定系统状态等问题的固有风险，我们引入了新的风险状态的概念来表示年龄度量高的状态。我们使用这个新的风险状态概念来量化和最小化高年龄度量的风险，通过直接导出风险状态的频率作为一种新的风险度量。在此基础上，我们引入了两种针对AoI，QAoI和AoII的风险敏感策略。第一种策略使用系统知识，

    Popular methods to quantify transmitted data quality are the Age of Information (AoI), the Query Age of Information (QAoI), and the Age of Incorrect Information (AoII). We consider these metrics in a point-to-point wireless communication system, where the transmitter monitors a process and sends status updates to a receiver. The challenge is to decide on the best time for an update, balancing the transmission energy and the age-based metric at the receiver. Due to the inherent risk of high age-based metric values causing complications such as unstable system states, we introduce the new concept of risky states to denote states with high age-based metric. We use this new notion of risky states to quantify and minimize this risk of experiencing high age-based metrics by directly deriving the frequency of risky states as a novel risk-metric. Building on this foundation, we introduce two risk-sensitive strategies for AoI, QAoI and AoII. The first strategy uses system knowledge, i.e., chann
    
[^3]: Nystr\"om $M$-Hilbert-Schmidt独立准则

    Nystr\"om $M$-Hilbert-Schmidt Independence Criterion. (arXiv:2302.09930v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.09930](http://arxiv.org/abs/2302.09930)

    这项研究提出了Nystr\"om $M$-Hilbert-Schmidt独立准则，针对大规模应用的二次计算瓶颈问题进行了解决，并兼顾了多个随机变量的推广情况和理论保证。

    

    核技术是数据科学中最受欢迎和强大的方法之一。核的广泛应用的关键特性包括：(i) 它们针对的领域数量多，(ii) 与核相关的函数类具有Hilbert结构，便于统计分析，以及(iii) 它们能够以不丢失信息的方式表示概率分布。这些特性导致了Hilbert-Schmidt独立准则(HSIC)的巨大成功，该准则能够在温和条件下捕捉随机变量的联合独立性，并允许具有二次计算复杂性的闭式估计器(相对于样本大小)。为了解决大规模应用中的二次计算瓶颈问题，已经提出了多个HSIC近似估计器，然而这些估计器限制于$M=2$个随机变量，不能自然地推广到$M \geq 2$的情况，并且缺乏理论保证。在这项工作中，我们提出了一个Nystr\"om $M$-Hilbert-Schmidt独立准则来解决这个问题。

    Kernel techniques are among the most popular and powerful approaches of data science. Among the key features that make kernels ubiquitous are (i) the number of domains they have been designed for, (ii) the Hilbert structure of the function class associated to kernels facilitating their statistical analysis, and (iii) their ability to represent probability distributions without loss of information. These properties give rise to the immense success of Hilbert-Schmidt independence criterion (HSIC) which is able to capture joint independence of random variables under mild conditions, and permits closed-form estimators with quadratic computational complexity (w.r.t. the sample size). In order to alleviate the quadratic computational bottleneck in large-scale applications, multiple HSIC approximations have been proposed, however these estimators are restricted to $M=2$ random variables, do not extend naturally to the $M\ge 2$ case, and lack theoretical guarantees. In this work, we propose an
    

