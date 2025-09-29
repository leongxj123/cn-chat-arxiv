# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Leveraging World Model Disentanglement in Value-Based Multi-Agent Reinforcement Learning.](http://arxiv.org/abs/2309.04615) | 本文提出了一种新的基于模型的多智能体强化学习方法，通过使用模块化的世界模型，减少了多智能体系统中训练的样本复杂性，并成功预测了联合动作价值函数。 |

# 详细

[^1]: 利用世界模型分解在基于值的多智能体强化学习中的应用

    Leveraging World Model Disentanglement in Value-Based Multi-Agent Reinforcement Learning. (arXiv:2309.04615v1 [cs.LG])

    [http://arxiv.org/abs/2309.04615](http://arxiv.org/abs/2309.04615)

    本文提出了一种新的基于模型的多智能体强化学习方法，通过使用模块化的世界模型，减少了多智能体系统中训练的样本复杂性，并成功预测了联合动作价值函数。

    

    本文提出了一种新颖的基于模型的多智能体强化学习方法，名为Value Decomposition Framework with Disentangled World Model，旨在解决在相同环境中多个智能体达成共同目标时的样本复杂性问题。由于多智能体系统的可扩展性和非平稳性问题，无模型方法依赖于大量样本进行训练。相反地，我们使用模块化的世界模型，包括动作条件、无动作和静态分支，来解开环境动态并根据过去的经验产生想象中的结果，而不是直接从真实环境中采样。我们使用变分自动编码器和变分图自动编码器来学习世界模型的潜在表示，将其与基于值的框架合并，以预测联合动作价值函数并优化整体训练目标。我们提供实验结果。

    In this paper, we propose a novel model-based multi-agent reinforcement learning approach named Value Decomposition Framework with Disentangled World Model to address the challenge of achieving a common goal of multiple agents interacting in the same environment with reduced sample complexity. Due to scalability and non-stationarity problems posed by multi-agent systems, model-free methods rely on a considerable number of samples for training. In contrast, we use a modularized world model, composed of action-conditioned, action-free, and static branches, to unravel the environment dynamics and produce imagined outcomes based on past experience, without sampling directly from the real environment. We employ variational auto-encoders and variational graph auto-encoders to learn the latent representations for the world model, which is merged with a value-based framework to predict the joint action-value function and optimize the overall training objective. We present experimental results 
    

