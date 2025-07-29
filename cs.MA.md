# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Contrastive learning-based agent modeling for deep reinforcement learning.](http://arxiv.org/abs/2401.00132) | 本研究提出了一种基于对比学习的深度强化学习代理建模方法，该方法可以在仅利用自我代理的本地观测的情况下，提取其他代理的有意义策略表示，以改进自我代理的自适应策略。 |

# 详细

[^1]: 基于对比学习的深度强化学习代理建模

    Contrastive learning-based agent modeling for deep reinforcement learning. (arXiv:2401.00132v2 [cs.MA] UPDATED)

    [http://arxiv.org/abs/2401.00132](http://arxiv.org/abs/2401.00132)

    本研究提出了一种基于对比学习的深度强化学习代理建模方法，该方法可以在仅利用自我代理的本地观测的情况下，提取其他代理的有意义策略表示，以改进自我代理的自适应策略。

    

    多智能体系统经常需要代理与具有不同目标、行为或策略的其他代理合作或竞争。在多智能体系统中设计自适应策略时，代理建模是必不可少的，因为这是自我代理理解其他代理行为并提取有意义的策略表示的方式。这些表示可以用来增强自我代理的自适应策略，该策略通过强化学习进行训练。然而，现有的代理建模方法通常假设在训练或长时间观察轨迹的策略适应过程中可以使用来自其他代理（建模代理）的本地观测。为了消除这些限制性假设并提高代理建模性能，我们设计了一种基于对比学习的代理建模（CLAM）方法，该方法仅依赖于自我代理在训练和执行过程中的本地观测。

    Multi-agent systems often require agents to collaborate with or compete against other agents with diverse goals, behaviors, or strategies. Agent modeling is essential when designing adaptive policies for intelligent machine agents in multiagent systems, as this is the means by which the ego agent understands other agents' behavior and extracts their meaningful policy representations. These representations can be used to enhance the ego agent's adaptive policy which is trained by reinforcement learning. However, existing agent modeling approaches typically assume the availability of local observations from other agents (modeled agents) during training or a long observation trajectory for policy adaption. To remove these constrictive assumptions and improve agent modeling performance, we devised a Contrastive Learning-based Agent Modeling (CLAM) method that relies only on the local observations from the ego agent during training and execution. With these observations, CLAM is capable of 
    

