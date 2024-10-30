# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Interpretable Concept Bottlenecks to Align Reinforcement Learning Agents.](http://arxiv.org/abs/2401.05821) | SCoBots是一种可解释的概念瓶颈代理，能够透明化整个决策流程，帮助领域专家理解和规范强化学习代理的行为，从而可能实现更好的人类对齐强化学习。 |

# 详细

[^1]: 可解释概念瓶颈用于对齐强化学习代理

    Interpretable Concept Bottlenecks to Align Reinforcement Learning Agents. (arXiv:2401.05821v1 [cs.LG])

    [http://arxiv.org/abs/2401.05821](http://arxiv.org/abs/2401.05821)

    SCoBots是一种可解释的概念瓶颈代理，能够透明化整个决策流程，帮助领域专家理解和规范强化学习代理的行为，从而可能实现更好的人类对齐强化学习。

    

    奖励稀疏性、难以归因的问题以及不对齐等等都是深度强化学习代理学习最优策略困难甚至不可能的原因之一。不幸的是，深度网络的黑盒特性阻碍了领域专家的参与，这些专家可以解释模型并纠正错误行为。为此，我们引入了连续概念瓶颈代理（SCoBots），通过整合连续的概念瓶颈层，使整个决策流程透明化。SCoBots不仅利用相关的对象属性，还利用关系概念。实验结果强有力地证明，SCoBots使领域专家能够有效理解和规范他们的行为，从而可能实现更好的人类对齐强化学习。通过这种方式，SCoBots使我们能够识别最简单且具有代表性的视频游戏Pong中的不对齐问题，并加以解决。

    Reward sparsity, difficult credit assignment, and misalignment are only a few of the many issues that make it difficult, if not impossible, for deep reinforcement learning (RL) agents to learn optimal policies. Unfortunately, the black-box nature of deep networks impedes the inclusion of domain experts who could interpret the model and correct wrong behavior. To this end, we introduce Successive Concept Bottlenecks Agents (SCoBots), which make the whole decision pipeline transparent via the integration of consecutive concept bottleneck layers. SCoBots make use of not only relevant object properties but also of relational concepts. Our experimental results provide strong evidence that SCoBots allow domain experts to efficiently understand and regularize their behavior, resulting in potentially better human-aligned RL. In this way, SCoBots enabled us to identify a misalignment problem in the most simple and iconic video game, Pong, and resolve it.
    

