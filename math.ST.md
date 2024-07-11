# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tuning-free testing of factor regression against factor-augmented sparse alternatives.](http://arxiv.org/abs/2307.13364) | 该研究提出了一种不需要调参、不需要估计协方差矩阵的自助法测试方法，用于评估因子回归模型在高维因子增强稀疏回归模型中的适用性。通过模拟实验证明了该方法的良好性能，并在应用于实际数据集时拒绝了经典因子回归模型的拟合性。 |

# 详细

[^1]: 不需要调参的因子回归测试与因子增强稀疏的对立. (arXiv:2307.13364v1 [econ.EM])

    Tuning-free testing of factor regression against factor-augmented sparse alternatives. (arXiv:2307.13364v1 [econ.EM])

    [http://arxiv.org/abs/2307.13364](http://arxiv.org/abs/2307.13364)

    该研究提出了一种不需要调参、不需要估计协方差矩阵的自助法测试方法，用于评估因子回归模型在高维因子增强稀疏回归模型中的适用性。通过模拟实验证明了该方法的良好性能，并在应用于实际数据集时拒绝了经典因子回归模型的拟合性。

    

    该研究引入了一种基于自助法的因子回归有效性测试方法，该方法在集成了因子回归和稀疏回归技术的高维因子增强稀疏回归模型中使用。该测试方法提供了一种评估经典（密集）因子回归模型与替代（稀疏加密集）因子增强稀疏回归模型的适用性的方式。我们提出的测试不需要调参，消除了估计协方差矩阵的需求，并且在实现上简单。该测试的有效性在时间序列相关性下在理论上得到了证明。通过模拟实验，我们展示了我们的方法在有限样本下的良好性能。此外，我们使用FRED-MD数据集应用了该测试，并在因变量为通胀时拒绝了经典因子回归模型的拟合性，但在因变量为工业生产时未拒绝。这些发现为选择适当模型提供了参考。

    This study introduces a bootstrap test of the validity of factor regression within a high-dimensional factor-augmented sparse regression model that integrates factor and sparse regression techniques. The test provides a means to assess the suitability of the classical (dense) factor regression model compared to alternative (sparse plus dense) factor-augmented sparse regression models. Our proposed test does not require tuning parameters, eliminates the need to estimate covariance matrices, and offers simplicity in implementation. The validity of the test is theoretically established under time-series dependence. Through simulation experiments, we demonstrate the favorable finite sample performance of our procedure. Moreover, using the FRED-MD dataset, we apply the test and reject the adequacy of the classical factor regression model when the dependent variable is inflation but not when it is industrial production. These findings offer insights into selecting appropriate models for high
    

