# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PoCo: Policy Composition from and for Heterogeneous Robot Learning](https://arxiv.org/abs/2402.02511) | PoCo是一种策略组合方法，通过组合不同模态和领域的数据分布，实现了从异构数据中训练通用机器人策略的目标。该方法可以实现场景级和任务级的广义操作技能学习，并在推理时通过任务级组合和分析成本函数进行策略行为的自适应调整。 |

# 详细

[^1]: PoCo: 来自和为异构机器人学习的策略组合

    PoCo: Policy Composition from and for Heterogeneous Robot Learning

    [https://arxiv.org/abs/2402.02511](https://arxiv.org/abs/2402.02511)

    PoCo是一种策略组合方法，通过组合不同模态和领域的数据分布，实现了从异构数据中训练通用机器人策略的目标。该方法可以实现场景级和任务级的广义操作技能学习，并在推理时通过任务级组合和分析成本函数进行策略行为的自适应调整。

    

    从异构数据中训练通用的机器人策略以处理不同任务是一个重大挑战。现有的机器人数据集在颜色、深度、触觉和姿态感等不同模态上存在差异，并在模拟、真实机器人和人类视频等不同领域收集。目前的方法通常会收集并汇集一个领域的所有数据，以训练一个单一策略来处理任务和领域的异构性，这是非常昂贵和困难的。在这项工作中，我们提出了一种灵活的方法，称为策略组合，通过组合用扩散模型表示的不同数据分布，将跨不同模态和领域的信息结合起来，以学习场景级和任务级的广义操作技能。我们的方法可以在多任务操作中使用任务级组合，并与分析成本函数组合，以在推理时调整策略行为。我们在模拟、人类和实际机器人数据上训练我们的方法。

    Training general robotic policies from heterogeneous data for different tasks is a significant challenge. Existing robotic datasets vary in different modalities such as color, depth, tactile, and proprioceptive information, and collected in different domains such as simulation, real robots, and human videos. Current methods usually collect and pool all data from one domain to train a single policy to handle such heterogeneity in tasks and domains, which is prohibitively expensive and difficult. In this work, we present a flexible approach, dubbed Policy Composition, to combine information across such diverse modalities and domains for learning scene-level and task-level generalized manipulation skills, by composing different data distributions represented with diffusion models. Our method can use task-level composition for multi-task manipulation and be composed with analytic cost functions to adapt policy behaviors at inference time. We train our method on simulation, human, and real 
    

