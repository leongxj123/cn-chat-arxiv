# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Experiential Explanations for Reinforcement Learning.](http://arxiv.org/abs/2210.04723) | 该论文提出了一种经验解释技术，通过训练影响预测器来恢复强化学习系统中的信息，使得非AI专家能够更好地理解其决策过程。 |

# 详细

[^1]: 强化学习的经验解释

    Experiential Explanations for Reinforcement Learning. (arXiv:2210.04723v3 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2210.04723](http://arxiv.org/abs/2210.04723)

    该论文提出了一种经验解释技术，通过训练影响预测器来恢复强化学习系统中的信息，使得非AI专家能够更好地理解其决策过程。

    

    强化学习系统可能非常复杂和无法解释，这使得非人工智能专家难以理解或干预它们的决策。我们提出了一种经验解释技术，通过在强化学习策略旁边训练影响预测器来生成反事实解释。影响预测器是学习奖励来源如何影响代理在不同状态下的模型，从而恢复有关策略如何反映环境的信息。

    Reinforcement Learning (RL) systems can be complex and non-interpretable, making it challenging for non-AI experts to understand or intervene in their decisions. This is due, in part, to the sequential nature of RL in which actions are chosen because of future rewards. However, RL agents discard the qualitative features of their training, making it hard to recover user-understandable information for "why" an action is chosen. Proposed sentence chunking: We propose a technique Experiential Explanations to generate counterfactual explanations by training influence predictors alongside the RL policy. Influence predictors are models that learn how sources of reward affect the agent in different states, thus restoring information about how the policy reflects the environment. A human evaluation study revealed that participants presented with experiential explanations were better able to correctly guess what an agent would do than those presented with other standard types of explanations. Pa
    

