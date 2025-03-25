# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Limits to Reservoir Learning.](http://arxiv.org/abs/2307.14474) | 这项工作限制了机器学习的能力，基于物理学所暗示的计算限制。储水库计算机在噪声下的性能下降意味着需要指数数量的样本来学习函数族，并讨论了没有噪声时的性能。 |

# 详细

[^1]: 河川学习的限制。

    Limits to Reservoir Learning. (arXiv:2307.14474v1 [cs.LG])

    [http://arxiv.org/abs/2307.14474](http://arxiv.org/abs/2307.14474)

    这项工作限制了机器学习的能力，基于物理学所暗示的计算限制。储水库计算机在噪声下的性能下降意味着需要指数数量的样本来学习函数族，并讨论了没有噪声时的性能。

    

    在这项工作中，我们根据物理学所暗示的计算限制来限制机器学习的能力。我们首先考虑信息处理能力（IPC），这是一个对信号集合到完整函数基的期望平方误差进行归一化的指标。我们使用IPC来衡量噪声下储水库计算机（一种特殊的循环网络）的性能降低。首先，我们证明IPC在系统尺寸n上是一个多项式，即使考虑到n个输出信号的$2^n$个可能的逐点乘积。接下来，我们认为这种退化意味着在储水库噪声存在的情况下，储水库所表示的函数族需要指数数量的样本来进行学习。最后，我们讨论了在没有噪声的情况下，同一集合的$2^n$个函数在进行二元分类时的性能。

    In this work, we bound a machine's ability to learn based on computational limitations implied by physicality. We start by considering the information processing capacity (IPC), a normalized measure of the expected squared error of a collection of signals to a complete basis of functions. We use the IPC to measure the degradation under noise of the performance of reservoir computers, a particular kind of recurrent network, when constrained by physical considerations. First, we show that the IPC is at most a polynomial in the system size $n$, even when considering the collection of $2^n$ possible pointwise products of the $n$ output signals. Next, we argue that this degradation implies that the family of functions represented by the reservoir requires an exponential number of samples to learn in the presence of the reservoir's noise. Finally, we conclude with a discussion of the performance of the same collection of $2^n$ functions without noise when being used for binary classification
    

