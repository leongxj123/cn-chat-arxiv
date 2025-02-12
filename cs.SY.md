# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Simple Way to Incorporate Novelty Detection in World Models.](http://arxiv.org/abs/2310.08731) | 本文提出了一个简单的方法，通过利用世界模型幻觉状态和真实观察状态的不匹配性作为异常分数，将新颖性检测纳入世界模型强化学习代理中。在新环境中与传统方法相比，我们的工作具有优势。 |

# 详细

[^1]: 一个简单的方法在世界模型中实现新颖性检测

    A Simple Way to Incorporate Novelty Detection in World Models. (arXiv:2310.08731v1 [cs.AI])

    [http://arxiv.org/abs/2310.08731](http://arxiv.org/abs/2310.08731)

    本文提出了一个简单的方法，通过利用世界模型幻觉状态和真实观察状态的不匹配性作为异常分数，将新颖性检测纳入世界模型强化学习代理中。在新环境中与传统方法相比，我们的工作具有优势。

    

    使用世界模型进行强化学习已经取得了显著的成功。然而，当世界机制或属性发生突然变化时，代理的性能和可靠性可能会显著下降。我们将视觉属性或状态转换的突变称为“新颖性”。在生成的世界模型框架中实施新颖性检测是保护部署时代理的关键任务。在本文中，我们提出了一种简单的边界方法，用于将新颖性检测纳入世界模型强化学习代理中，通过利用世界模型幻觉状态和真实观察状态的不匹配性作为异常分数。首先，我们提供了与序列决策相关的新颖性检测本体论，然后我们提供了在代理在世界模型中学习的转换分布中检测新颖性的有效方法。最后，我们展示了我们的工作在新环境中与传统方法相比的优势。

    Reinforcement learning (RL) using world models has found significant recent successes. However, when a sudden change to world mechanics or properties occurs then agent performance and reliability can dramatically decline. We refer to the sudden change in visual properties or state transitions as {\em novelties}. Implementing novelty detection within generated world model frameworks is a crucial task for protecting the agent when deployed. In this paper, we propose straightforward bounding approaches to incorporate novelty detection into world model RL agents, by utilizing the misalignment of the world model's hallucinated states and the true observed states as an anomaly score. We first provide an ontology of novelty detection relevant to sequential decision making, then we provide effective approaches to detecting novelties in a distribution of transitions learned by an agent in a world model. Finally, we show the advantage of our work in a novel environment compared to traditional ma
    

