# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Guided Decoding for Robot Motion Generation and Adaption](https://arxiv.org/abs/2403.15239) | 通过将演示学习集成到运动生成中，使机器人能够实时生成适应复杂环境的运动 |
| [^2] | [Foundation Reinforcement Learning: towards Embodied Generalist Agents with Foundation Prior Assistance.](http://arxiv.org/abs/2310.02635) | 本研究提出了一种基于具身基础先验的基础强化学习框架，通过加速训练过程来提高样本效率。 |

# 详细

[^1]: 引导解码用于机器人运动生成和适应

    Guided Decoding for Robot Motion Generation and Adaption

    [https://arxiv.org/abs/2403.15239](https://arxiv.org/abs/2403.15239)

    通过将演示学习集成到运动生成中，使机器人能够实时生成适应复杂环境的运动

    

    我们针对具有障碍物、通过点等复杂环境下的高自由度机器人臂运动生成问题进行了探讨。通过将演示学习（LfD）集成到运动生成过程中，取得了该领域的重大进展。这种集成支持机器人快速适应新任务，并通过允许机器人从演示轨迹中学习和泛化来优化积累的经验利用。我们在大量模拟轨迹数据集上训练了一个变分自动编码器变换器的transformer架构。这种基于条件变分自动编码器变换器的架构学习了基本的运动生成技能，并将其调整以满足辅助任务和约束条件。我们的自回归方法实现了物理系统反馈的实时集成，增强了运动生成的适应性和效率。我们展示了我们的模型能够从初始点和目标点生成运动，同时

    arXiv:2403.15239v1 Announce Type: cross  Abstract: We address motion generation for high-DoF robot arms in complex settings with obstacles, via points, etc. A significant advancement in this domain is achieved by integrating Learning from Demonstration (LfD) into the motion generation process. This integration facilitates rapid adaptation to new tasks and optimizes the utilization of accumulated expertise by allowing robots to learn and generalize from demonstrated trajectories.   We train a transformer architecture on a large dataset of simulated trajectories. This architecture, based on a conditional variational autoencoder transformer, learns essential motion generation skills and adapts these to meet auxiliary tasks and constraints. Our auto-regressive approach enables real-time integration of feedback from the physical system, enhancing the adaptability and efficiency of motion generation. We show that our model can generate motion from initial and target points, but also that it 
    
[^2]: 强化学习基础：朝向具有基础先验辅助的具身通用智能体

    Foundation Reinforcement Learning: towards Embodied Generalist Agents with Foundation Prior Assistance. (arXiv:2310.02635v1 [cs.RO])

    [http://arxiv.org/abs/2310.02635](http://arxiv.org/abs/2310.02635)

    本研究提出了一种基于具身基础先验的基础强化学习框架，通过加速训练过程来提高样本效率。

    

    最近人们已经表明，从互联网规模的数据中进行大规模预训练是构建通用模型的关键，正如在NLP中所见。为了构建具身通用智能体，我们和许多其他研究者假设这种基础先验也是不可或缺的组成部分。然而，目前尚不清楚如何以适当的具体形式表示这些具身基础先验，以及它们应该如何在下游任务中使用。在本文中，我们提出了一组直观有效的具身先验，包括基础策略、价值和成功奖励。所提出的先验是基于目标条件的MDP。为了验证其有效性，我们实例化了一个由这些先验辅助的演员-评论家方法，称之为基础演员-评论家（FAC）。我们将我们的框架命名为基础强化学习（FRL），因为它完全依赖于具身基础先验来进行探索、学习和强化。FRL的好处有三个。(1)样本效率高。通过基础先验加速训练过程，减少样本使用量。

    Recently, people have shown that large-scale pre-training from internet-scale data is the key to building generalist models, as witnessed in NLP. To build embodied generalist agents, we and many other researchers hypothesize that such foundation prior is also an indispensable component. However, it is unclear what is the proper concrete form to represent those embodied foundation priors and how they should be used in the downstream task. In this paper, we propose an intuitive and effective set of embodied priors that consist of foundation policy, value, and success reward. The proposed priors are based on the goal-conditioned MDP. To verify their effectiveness, we instantiate an actor-critic method assisted by the priors, called Foundation Actor-Critic (FAC). We name our framework as Foundation Reinforcement Learning (FRL), since it completely relies on embodied foundation priors to explore, learn and reinforce. The benefits of FRL are threefold. (1) Sample efficient. With foundation p
    

