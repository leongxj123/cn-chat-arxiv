# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adapting tree-based multiple imputation methods for multi-level data? A simulation study.](http://arxiv.org/abs/2401.14161) | 该研究通过模拟实验比较了传统的多重插补与基于树的方法在多层数据上的性能，发现MICE在准确的拒绝率方面优于其他方法，而极限梯度提升在减少偏差方面表现较好。 |

# 详细

[^1]: 适应多层数据的基于树的多重插补方法的研究

    Adapting tree-based multiple imputation methods for multi-level data? A simulation study. (arXiv:2401.14161v1 [stat.AP])

    [http://arxiv.org/abs/2401.14161](http://arxiv.org/abs/2401.14161)

    该研究通过模拟实验比较了传统的多重插补与基于树的方法在多层数据上的性能，发现MICE在准确的拒绝率方面优于其他方法，而极限梯度提升在减少偏差方面表现较好。

    

    本模拟研究评估了针对多层数据的多重插补(MI)技术的有效性。它比较了传统的以链式方程为基础的多重插补(MICE)与基于树的方法（如链式随机森林与预测均值匹配和极限梯度提升）的性能。还对基于树的方法包括了包括集群成员的虚拟变量的改进版本进行了评估。该研究使用具有不同集群大小(25和50)和不完整程度(10\%和50\%)的模拟分层数据对系数估计偏差、统计功效和类型I错误率进行评估。系数是使用随机截距和随机斜率模型进行估计的。结果表明，虽然MICE更适合准确的拒绝率，但极限梯度提升有助于减少偏差。此外，研究发现，不同集群大小的偏差水平相似，但拒绝率在少数缺失情况下较不理想。

    This simulation study evaluates the effectiveness of multiple imputation (MI) techniques for multilevel data. It compares the performance of traditional Multiple Imputation by Chained Equations (MICE) with tree-based methods such as Chained Random Forests with Predictive Mean Matching and Extreme Gradient Boosting. Adapted versions that include dummy variables for cluster membership are also included for the tree-based methods. Methods are evaluated for coefficient estimation bias, statistical power, and type I error rates on simulated hierarchical data with different cluster sizes (25 and 50) and levels of missingness (10\% and 50\%). Coefficients are estimated using random intercept and random slope models. The results show that while MICE is preferred for accurate rejection rates, Extreme Gradient Boosting is advantageous for reducing bias. Furthermore, the study finds that bias levels are similar across different cluster sizes, but rejection rates tend to be less favorable with few
    

