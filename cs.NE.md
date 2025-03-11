# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How Much is Unseen Depends Chiefly on Information About the Seen](https://arxiv.org/abs/2402.05835) | 该论文发现，在未知种群中属于未在训练数据中出现的类的数据点的比例几乎完全取决于训练数据中出现相同次数的类的数量。论文提出了一个遗传算法，能够根据样本找到一个具有最小均方误差的估计量。 |

# 详细

[^1]: 不可见数据取决于已知信息的多少

    How Much is Unseen Depends Chiefly on Information About the Seen

    [https://arxiv.org/abs/2402.05835](https://arxiv.org/abs/2402.05835)

    该论文发现，在未知种群中属于未在训练数据中出现的类的数据点的比例几乎完全取决于训练数据中出现相同次数的类的数量。论文提出了一个遗传算法，能够根据样本找到一个具有最小均方误差的估计量。

    

    乍一看可能有些违反直觉：我们发现，在预期中，未知种群中属于在训练数据中没有出现的类的数据点的比例几乎完全由训练数据中出现相同次数的类的数量$f_k$确定。虽然在理论上我们证明了由该估计量引起的偏差在样本大小指数级衰减，但在实践中，高方差阻止我们直接使用它作为样本覆盖估计量。但是，我们对$f_k$之间的依赖关系进行了精确的描述，从而产生了多个不同期望值表示的搜索空间，可以确定地实例化为估计量。因此，我们转向优化，并开发了一种遗传算法，仅根据样本搜索平均均方误差（MSE）最小的估计量。在我们的实验证明，我们的遗传算法发现了具有明显较小方差的估计量。

    It might seem counter-intuitive at first: We find that, in expectation, the proportion of data points in an unknown population-that belong to classes that do not appear in the training data-is almost entirely determined by the number $f_k$ of classes that do appear in the training data the same number of times. While in theory we show that the difference of the induced estimator decays exponentially in the size of the sample, in practice the high variance prevents us from using it directly for an estimator of the sample coverage. However, our precise characterization of the dependency between $f_k$'s induces a large search space of different representations of the expected value, which can be deterministically instantiated as estimators. Hence, we turn to optimization and develop a genetic algorithm that, given only the sample, searches for an estimator with minimal mean-squared error (MSE). In our experiments, our genetic algorithm discovers estimators that have a substantially smalle
    

