# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reconciling Spatial and Temporal Abstractions for Goal Representation.](http://arxiv.org/abs/2401.09870) | 本文介绍了一种新的三层分层强化学习算法，引入了空间和时间目标抽象化。研究者提供了学习策略的理论遗憾边界，并在多个任务上对算法进行了评估。 |

# 详细

[^1]: 调和空间和时间抽象化以实现目标表示

    Reconciling Spatial and Temporal Abstractions for Goal Representation. (arXiv:2401.09870v1 [cs.LG])

    [http://arxiv.org/abs/2401.09870](http://arxiv.org/abs/2401.09870)

    本文介绍了一种新的三层分层强化学习算法，引入了空间和时间目标抽象化。研究者提供了学习策略的理论遗憾边界，并在多个任务上对算法进行了评估。

    

    目标表示通过将复杂的学习问题分解为更容易的子任务来影响分层强化学习算法的性能。最近的研究表明，保留时间抽象环境动态的表示方法在解决困难问题和提供优化理论保证方面是成功的。然而，这些方法在环境动态越来越复杂（即时间抽象转换关系依赖更多变量）的任务中无法扩展。另一方面，其他方法则尝试使用空间抽象来缓解前面的问题。它们的限制包括无法适应高维环境和对先前知识的依赖。本文提出了一种新的三层分层强化学习算法，分层结构的不同层次引入了空间和时间目标抽象化。我们对学习策略的遗憾边界进行了理论研究。我们评估了我们提出的算法在不同任务上的性能。

    Goal representation affects the performance of Hierarchical Reinforcement Learning (HRL) algorithms by decomposing the complex learning problem into easier subtasks. Recent studies show that representations that preserve temporally abstract environment dynamics are successful in solving difficult problems and provide theoretical guarantees for optimality. These methods however cannot scale to tasks where environment dynamics increase in complexity i.e. the temporally abstract transition relations depend on larger number of variables. On the other hand, other efforts have tried to use spatial abstraction to mitigate the previous issues. Their limitations include scalability to high dimensional environments and dependency on prior knowledge.  In this paper, we propose a novel three-layer HRL algorithm that introduces, at different levels of the hierarchy, both a spatial and a temporal goal abstraction. We provide a theoretical study of the regret bounds of the learned policies. We evalua
    

