# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Predictive Inference in Multi-environment Scenarios](https://arxiv.org/abs/2403.16336) | 本研究提出了在多环境预测问题中构建有效置信区间和置信集的方法，并展示了一种新的调整方法以适应问题难度，从而减少预测集大小，这在神经感应和物种分类数据集中的实际表现中得到验证。 |
| [^2] | [Selecting Penalty Parameters of High-Dimensional M-Estimators using Bootstrapping after Cross-Validation.](http://arxiv.org/abs/2104.04716) | 本论文提出了一种称为交叉验证后的自助法的新方法，用于选择高维M-估计器的惩罚参数。通过模拟实验证明，该方法在估计误差方面优于交叉验证，并在推断方面表现出色。在案例研究中，我们对Fryer Jr (2019)的研究进行了再验证，并确认了他的发现。 |

# 详细

[^1]: 多环境场景中的预测推断

    Predictive Inference in Multi-environment Scenarios

    [https://arxiv.org/abs/2403.16336](https://arxiv.org/abs/2403.16336)

    本研究提出了在多环境预测问题中构建有效置信区间和置信集的方法，并展示了一种新的调整方法以适应问题难度，从而减少预测集大小，这在神经感应和物种分类数据集中的实际表现中得到验证。

    

    我们解决了在跨多个环境的预测问题中构建有效置信区间和置信集的挑战。我们研究了适用于这些问题的两种覆盖类型，扩展了Jackknife和分裂一致方法，展示了如何在这种非传统的层次数据生成场景中获得无分布覆盖。我们的贡献还包括对非实值响应设置的扩展，以及这些一般问题中预测推断的一致性理论。我们展示了一种新的调整方法，以适应问题难度，这适用于具有层次数据的预测推断的现有方法以及我们开发的方法；这通过神经化学感应和物种分类数据集评估了这些方法的实际性能。

    arXiv:2403.16336v1 Announce Type: cross  Abstract: We address the challenge of constructing valid confidence intervals and sets in problems of prediction across multiple environments. We investigate two types of coverage suitable for these problems, extending the jackknife and split-conformal methods to show how to obtain distribution-free coverage in such non-traditional, hierarchical data-generating scenarios. Our contributions also include extensions for settings with non-real-valued responses and a theory of consistency for predictive inference in these general problems. We demonstrate a novel resizing method to adapt to problem difficulty, which applies both to existing approaches for predictive inference with hierarchical data and the methods we develop; this reduces prediction set sizes using limited information from the test environment, a key to the methods' practical performance, which we evaluate through neurochemical sensing and species classification datasets.
    
[^2]: 使用交叉验证后的自助法选择高维M-估计器的惩罚参数

    Selecting Penalty Parameters of High-Dimensional M-Estimators using Bootstrapping after Cross-Validation. (arXiv:2104.04716v4 [math.ST] UPDATED)

    [http://arxiv.org/abs/2104.04716](http://arxiv.org/abs/2104.04716)

    本论文提出了一种称为交叉验证后的自助法的新方法，用于选择高维M-估计器的惩罚参数。通过模拟实验证明，该方法在估计误差方面优于交叉验证，并在推断方面表现出色。在案例研究中，我们对Fryer Jr (2019)的研究进行了再验证，并确认了他的发现。

    

    我们开发了一种新的方法，用于在高维情况下选择$\ell_1$-惩罚M-估计器的惩罚参数，我们称之为交叉验证后的自助法。我们推导了相应的$\ell_1$-惩罚M-估计器的收敛速度，以及在准则函数中不使用惩罚重新拟合前一估计器的非零参数的后-$\ell_1$-惩罚M-估计器的收敛速度。通过模拟实验证明，我们的方法在估计误差方面不被交叉验证所主导，并且在推断方面胜过交叉验证。作为示例，我们重新审视了Fryer Jr (2019)关于警察使用武力的种族差异的研究，并确认了他的发现。

    We develop a new method for selecting the penalty parameter for $\ell_1$-penalized M-estimators in high dimensions, which we refer to as bootstrapping after cross-validation. We derive rates of convergence for the corresponding $\ell_1$-penalized M-estimator and also for the post-$\ell_1$-penalized M-estimator, which refits the non-zero parameters of the former estimator without penalty in the criterion function. We demonstrate via simulations that our method is not dominated by cross-validation in terms of estimation errors and outperforms cross-validation in terms of inference. As an illustration, we revisit Fryer Jr (2019), who investigated racial differences in police use of force, and confirm his findings.
    

