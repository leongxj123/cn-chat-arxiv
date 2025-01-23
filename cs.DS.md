# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stochastic Submodular Bandits with Delayed Composite Anonymous Bandit Feedback.](http://arxiv.org/abs/2303.13604) | 本论文研究了具有随机次模收益和全赌徒延迟反馈的组合多臂赌博机问题，研究了三种延迟反馈模型并导出了后悔上限。研究结果表明，算法能够在考虑延迟组合匿名反馈时胜过其他全赌徒方法。 |

# 详细

[^1]: 带有延迟组合匿名赌徒反馈的随机次模赌博算法

    Stochastic Submodular Bandits with Delayed Composite Anonymous Bandit Feedback. (arXiv:2303.13604v1 [cs.LG])

    [http://arxiv.org/abs/2303.13604](http://arxiv.org/abs/2303.13604)

    本论文研究了具有随机次模收益和全赌徒延迟反馈的组合多臂赌博机问题，研究了三种延迟反馈模型并导出了后悔上限。研究结果表明，算法能够在考虑延迟组合匿名反馈时胜过其他全赌徒方法。

    

    本文研究了组合多臂赌博机问题，其中包含了期望下的随机次模收益和全赌徒延迟反馈，延迟反馈被假定为组合和匿名。也就是说，延迟反馈是由过去行动的奖励组成的，这些奖励由子组件构成，其未知的分配方式。研究了三种延迟反馈模型：有界对抗模型、随机独立模型和随机条件独立模型，并针对每种延迟模型导出了后悔界。忽略问题相关参数，我们证明了所有延迟模型的后悔界为 $\tilde{O}(T^{2/3} + T^{1/3} \nu)$，其中 $T$ 是时间范围，$\nu$ 是三种情况下不同定义的延迟参数，因此展示了带有延迟的补偿项。所考虑的算法被证明能够胜过其他考虑了延迟组合匿名反馈的全赌徒方法。

    This paper investigates the problem of combinatorial multiarmed bandits with stochastic submodular (in expectation) rewards and full-bandit delayed feedback, where the delayed feedback is assumed to be composite and anonymous. In other words, the delayed feedback is composed of components of rewards from past actions, with unknown division among the sub-components. Three models of delayed feedback: bounded adversarial, stochastic independent, and stochastic conditionally independent are studied, and regret bounds are derived for each of the delay models. Ignoring the problem dependent parameters, we show that regret bound for all the delay models is $\tilde{O}(T^{2/3} + T^{1/3} \nu)$ for time horizon $T$, where $\nu$ is a delay parameter defined differently in the three cases, thus demonstrating an additive term in regret with delay in all the three delay models. The considered algorithm is demonstrated to outperform other full-bandit approaches with delayed composite anonymous feedbac
    

