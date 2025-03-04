# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Asymptotic Behavior of Adversarial Training Estimator under $\ell_\infty$-Perturbation.](http://arxiv.org/abs/2401.15262) | 本文研究了在$\ell_\infty$-扰动下的对抗性训练，证明当真实参数为0时，对抗性训练估计器在该扰动下的极限分布可能在0处有一个正概率质量，提供了稀疏性恢复能力的理论保证，并提出了一种两步过程——自适应对抗性训练，可以进一步提高性能。 |
| [^2] | [Optimal Decision Rules Under Partial Identification.](http://arxiv.org/abs/2111.04926) | 本文研究了在部分识别下的最优决策规则问题，提出了在已知方差的正态误差情况下的有限样本最小最大遗憾决策规则和渐近最小最大遗憾的可行决策规则，并应用于回归不连续设置中政策资格截断点的问题。 |

# 详细

[^1]: 在$\ell_\infty$-扰动下对抗性训练估计器的渐近行为

    Asymptotic Behavior of Adversarial Training Estimator under $\ell_\infty$-Perturbation. (arXiv:2401.15262v1 [math.ST])

    [http://arxiv.org/abs/2401.15262](http://arxiv.org/abs/2401.15262)

    本文研究了在$\ell_\infty$-扰动下的对抗性训练，证明当真实参数为0时，对抗性训练估计器在该扰动下的极限分布可能在0处有一个正概率质量，提供了稀疏性恢复能力的理论保证，并提出了一种两步过程——自适应对抗性训练，可以进一步提高性能。

    

    对抗性训练被提出来抵御机器学习和统计模型中的对抗性攻击。本文重点研究了在$\ell_\infty$-扰动下的对抗性训练，这个问题最近引起了很多研究的关注。在广义线性模型中研究了对抗性训练估计器的渐近行为。结果表明，当真实参数为0时，对抗性训练估计器在$\ell_\infty$-扰动下的极限分布可能在0处有一个正概率质量，为相关的稀疏性恢复能力提供了理论保证。此外，提出了一种两步过程——自适应对抗性训练，可以进一步提高在$\ell_\infty$-扰动下的对抗性训练的性能。具体而言，所提出的过程可以实现渐近无偏性和变量选择一致性。通过数值实验展示了稀疏性恢复的能力。

    Adversarial training has been proposed to hedge against adversarial attacks in machine learning and statistical models. This paper focuses on adversarial training under $\ell_\infty$-perturbation, which has recently attracted much research attention. The asymptotic behavior of the adversarial training estimator is investigated in the generalized linear model. The results imply that the limiting distribution of the adversarial training estimator under $\ell_\infty$-perturbation could put a positive probability mass at $0$ when the true parameter is $0$, providing a theoretical guarantee of the associated sparsity-recovery ability. Alternatively, a two-step procedure is proposed -adaptive adversarial training, which could further improve the performance of adversarial training under $\ell_\infty$-perturbation. Specifically, the proposed procedure could achieve asymptotic unbiasedness and variable-selection consistency. Numerical experiments are conducted to show the sparsity-recovery a
    
[^2]: 在部分识别下的最优决策规则

    Optimal Decision Rules Under Partial Identification. (arXiv:2111.04926v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2111.04926](http://arxiv.org/abs/2111.04926)

    本文研究了在部分识别下的最优决策规则问题，提出了在已知方差的正态误差情况下的有限样本最小最大遗憾决策规则和渐近最小最大遗憾的可行决策规则，并应用于回归不连续设置中政策资格截断点的问题。

    

    本文考虑了一类统计决策问题，决策者必须在有限样本的基础上，在两种备选策略之间做出决策，以最大化社会福利。核心假设是潜在的、可能是无限维参数位于已知的凸集中，可能导致福利效应的部分识别。这些限制的一个例子是反事实结果函数的平滑性。作为主要理论结果，我在正态分布误差且方差已知的所有决策规则类中，推导出了一种有限样本的最小最大遗憾决策规则。当误差分布未知时，我得到了一种渐近最小最大遗憾的可行决策规则。我将我的结果应用于在回归不连续设置中是否改变政策资格截断点的问题，并在布基纳法索的学校建设项目的实证应用中进行了阐述。

    I consider a class of statistical decision problems in which the policy maker must decide between two alternative policies to maximize social welfare based on a finite sample. The central assumption is that the underlying, possibly infinite-dimensional parameter, lies in a known convex set, potentially leading to partial identification of the welfare effect. An example of such restrictions is the smoothness of counterfactual outcome functions. As the main theoretical result, I derive a finite-sample, exact minimax regret decision rule within the class of all decision rules under normal errors with known variance. When the error distribution is unknown, I obtain a feasible decision rule that is asymptotically minimax regret. I apply my results to the problem of whether to change a policy eligibility cutoff in a regression discontinuity setup, and illustrate them in an empirical application to a school construction program in Burkina Faso.
    

