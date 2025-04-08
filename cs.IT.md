# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Quantized Hierarchical Federated Learning: A Robust Approach to Statistical Heterogeneity](https://arxiv.org/abs/2403.01540) | 该算法结合了分层联邦学习中的梯度聚合和模型聚合，通过量化提高通信效率，表现出对统计异质性的鲁棒性。 |
| [^2] | [Limits to Reservoir Learning.](http://arxiv.org/abs/2307.14474) | 这项工作限制了机器学习的能力，基于物理学所暗示的计算限制。储水库计算机在噪声下的性能下降意味着需要指数数量的样本来学习函数族，并讨论了没有噪声时的性能。 |

# 详细

[^1]: 分层量化联邦学习：统计异质性的一种强大方法

    Quantized Hierarchical Federated Learning: A Robust Approach to Statistical Heterogeneity

    [https://arxiv.org/abs/2403.01540](https://arxiv.org/abs/2403.01540)

    该算法结合了分层联邦学习中的梯度聚合和模型聚合，通过量化提高通信效率，表现出对统计异质性的鲁棒性。

    

    本文介绍了一种新颖的分层联邦学习算法，该算法在多个集合中结合了量化以提高通信效率，并展示了对于统计异质性的弹性。与传统的分层联邦学习算法不同，我们的方法在集合内迭代中结合了梯度聚合和集合间迭代中的模型聚合。我们提供了一个全面的分析框架来评估其最优性差距和收敛速度，将这些方面与传统算法进行了比较。此外，我们开发了一个问题表述，以导出封闭形式的最优系统参数解。我们的研究结果表明，我们的算法在一系列参数上始终实现高学习精度，并且在具有异构数据分布的场景中明显优于其他分层算法。

    arXiv:2403.01540v1 Announce Type: new  Abstract: This paper presents a novel hierarchical federated learning algorithm within multiple sets that incorporates quantization for communication-efficiency and demonstrates resilience to statistical heterogeneity. Unlike conventional hierarchical federated learning algorithms, our approach combines gradient aggregation in intra-set iterations with model aggregation in inter-set iterations. We offer a comprehensive analytical framework to evaluate its optimality gap and convergence rate, comparing these aspects with those of conventional algorithms. Additionally, we develop a problem formulation to derive optimal system parameters in a closed-form solution. Our findings reveal that our algorithm consistently achieves high learning accuracy over a range of parameters and significantly outperforms other hierarchical algorithms, particularly in scenarios with heterogeneous data distributions.
    
[^2]: 河川学习的限制。

    Limits to Reservoir Learning. (arXiv:2307.14474v1 [cs.LG])

    [http://arxiv.org/abs/2307.14474](http://arxiv.org/abs/2307.14474)

    这项工作限制了机器学习的能力，基于物理学所暗示的计算限制。储水库计算机在噪声下的性能下降意味着需要指数数量的样本来学习函数族，并讨论了没有噪声时的性能。

    

    在这项工作中，我们根据物理学所暗示的计算限制来限制机器学习的能力。我们首先考虑信息处理能力（IPC），这是一个对信号集合到完整函数基的期望平方误差进行归一化的指标。我们使用IPC来衡量噪声下储水库计算机（一种特殊的循环网络）的性能降低。首先，我们证明IPC在系统尺寸n上是一个多项式，即使考虑到n个输出信号的$2^n$个可能的逐点乘积。接下来，我们认为这种退化意味着在储水库噪声存在的情况下，储水库所表示的函数族需要指数数量的样本来进行学习。最后，我们讨论了在没有噪声的情况下，同一集合的$2^n$个函数在进行二元分类时的性能。

    In this work, we bound a machine's ability to learn based on computational limitations implied by physicality. We start by considering the information processing capacity (IPC), a normalized measure of the expected squared error of a collection of signals to a complete basis of functions. We use the IPC to measure the degradation under noise of the performance of reservoir computers, a particular kind of recurrent network, when constrained by physical considerations. First, we show that the IPC is at most a polynomial in the system size $n$, even when considering the collection of $2^n$ possible pointwise products of the $n$ output signals. Next, we argue that this degradation implies that the family of functions represented by the reservoir requires an exponential number of samples to learn in the presence of the reservoir's noise. Finally, we conclude with a discussion of the performance of the same collection of $2^n$ functions without noise when being used for binary classification
    

