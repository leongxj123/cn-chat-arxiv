# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bayesian Optimization with Formal Safety Guarantees via Online Conformal Prediction.](http://arxiv.org/abs/2306.17815) | 本文提出了一种基于贝叶斯优化的方法，无论约束函数的特性如何，都能满足安全要求。 |

# 详细

[^1]: 通过在线信心预测实现具备形式安全保证的贝叶斯优化

    Bayesian Optimization with Formal Safety Guarantees via Online Conformal Prediction. (arXiv:2306.17815v1 [cs.LG])

    [http://arxiv.org/abs/2306.17815](http://arxiv.org/abs/2306.17815)

    本文提出了一种基于贝叶斯优化的方法，无论约束函数的特性如何，都能满足安全要求。

    

    黑盒零阶优化是金融、物理和工程等领域应用的核心基本操作。在这个问题的常见形式中，设计者顺序尝试候选解，并从系统中接收到关于每个尝试值的噪声反馈。本文研究了在这些场景中还提供了有关尝试解的安全性的反馈，并且优化器被限制在整个优化过程中尝试的不安全解的数量上。在基于贝叶斯优化（BO）的方法上，先前的研究引入了一种被称为SAFEOPT的优化方案，只要满足对安全约束函数的严格假设，就能够以可控的概率在反馈噪声上避免选择任何不安全的解。本文介绍了一种新的基于BO的方法，无论约束函数的特性如何，都能满足安全要求。

    Black-box zero-th order optimization is a central primitive for applications in fields as diverse as finance, physics, and engineering. In a common formulation of this problem, a designer sequentially attempts candidate solutions, receiving noisy feedback on the value of each attempt from the system. In this paper, we study scenarios in which feedback is also provided on the safety of the attempted solution, and the optimizer is constrained to limit the number of unsafe solutions that are tried throughout the optimization process. Focusing on methods based on Bayesian optimization (BO), prior art has introduced an optimization scheme -- referred to as SAFEOPT -- that is guaranteed not to select any unsafe solution with a controllable probability over feedback noise as long as strict assumptions on the safety constraint function are met. In this paper, a novel BO-based approach is introduced that satisfies safety requirements irrespective of properties of the constraint function. This s
    

