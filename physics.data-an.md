# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Misspecification uncertainties in near-deterministic regression](https://arxiv.org/abs/2402.01810) | 该论文研究了近确定性回归中错误规范化的不确定性问题，并提出了一种组合模型，以准确预测和控制参数不确定性。 |

# 详细

[^1]: 近确定性回归中的错误规范化不确定性

    Misspecification uncertainties in near-deterministic regression

    [https://arxiv.org/abs/2402.01810](https://arxiv.org/abs/2402.01810)

    该论文研究了近确定性回归中错误规范化的不确定性问题，并提出了一种组合模型，以准确预测和控制参数不确定性。

    

    期望损失是模型泛化误差的上界，可用于学习的鲁棒PAC-Bayes边界。然而，损失最小化被认为忽略了错误规范化，即模型不能完全复制观测结果。这导致大数据或欠参数化极限下对参数不确定性的显著低估。我们分析近确定性、错误规范化和欠参数化替代模型的泛化误差，这是科学和工程中广泛相关的一个领域。我们证明后验分布必须覆盖每个训练点，以避免发散的泛化误差，并导出一个符合这个约束的组合模型。对于线性模型，这种高效的方法产生的额外开销最小。这种高效方法在模型问题上进行了演示，然后应用于原子尺度机器学习中的高维数据集。

    The expected loss is an upper bound to the model generalization error which admits robust PAC-Bayes bounds for learning. However, loss minimization is known to ignore misspecification, where models cannot exactly reproduce observations. This leads to significant underestimates of parameter uncertainties in the large data, or underparameterized, limit. We analyze the generalization error of near-deterministic, misspecified and underparametrized surrogate models, a regime of broad relevance in science and engineering. We show posterior distributions must cover every training point to avoid a divergent generalization error and derive an ensemble {ansatz} that respects this constraint, which for linear models incurs minimal overhead. The efficient approach is demonstrated on model problems before application to high dimensional datasets in atomistic machine learning. Parameter uncertainties from misspecification survive in the underparametrized limit, giving accurate prediction and boundin
    

