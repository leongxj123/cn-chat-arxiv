# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Large-Batch, Neural Multi-Objective Bayesian Optimization.](http://arxiv.org/abs/2306.01095) | 本文提出了一种针对数据密集型问题和多目标优化设置的贝叶斯优化框架，该方法利用了贝叶斯神经网络代理建模和可扩展、具有不确定性的收购策略，能够在最少迭代次数的情况下高效地进行优化。 |

# 详细

[^1]: 大批量神经多目标贝叶斯优化

    Large-Batch, Neural Multi-Objective Bayesian Optimization. (arXiv:2306.01095v1 [cs.LG])

    [http://arxiv.org/abs/2306.01095](http://arxiv.org/abs/2306.01095)

    本文提出了一种针对数据密集型问题和多目标优化设置的贝叶斯优化框架，该方法利用了贝叶斯神经网络代理建模和可扩展、具有不确定性的收购策略，能够在最少迭代次数的情况下高效地进行优化。

    

    贝叶斯优化在全局优化黑盒高成本函数方面提供了强大的框架。然而，由于默认高斯过程代理的可扩展性差，它在处理数据密集型问题，特别是在多目标设置中的能力有限。本文提出了一种新颖的贝叶斯优化框架，专为解决这些限制而设计。我们的方法利用了贝叶斯神经网络方法进行代理建模。这使得它能够有效地处理大批量数据，建模复杂问题以及产生预测的不确定性。此外，我们的方法结合了一种基于众所周知且易于部署的NSGA-II的可扩展的、具有不确定性的收购策略。这种完全可并行化的策略促进了未勘探区域的有效探索。我们的框架允许在最少迭代次数的情况下在数据密集环境中进行有效的优化。我们展示了我们方法的优越性。

    Bayesian optimization provides a powerful framework for global optimization of black-box, expensive-to-evaluate functions. However, it has a limited capacity in handling data-intensive problems, especially in multi-objective settings, due to the poor scalability of default Gaussian Process surrogates. We present a novel Bayesian optimization framework specifically tailored to address these limitations. Our method leverages a Bayesian neural networks approach for surrogate modeling. This enables efficient handling of large batches of data, modeling complex problems, and generating the uncertainty of the predictions. In addition, our method incorporates a scalable, uncertainty-aware acquisition strategy based on the well-known, easy-to-deploy NSGA-II. This fully parallelizable strategy promotes efficient exploration of uncharted regions. Our framework allows for effective optimization in data-intensive environments with a minimum number of iterations. We demonstrate the superiority of ou
    

