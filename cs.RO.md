# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diffusion-Reinforcement Learning Hierarchical Motion Planning in Adversarial Multi-agent Games](https://arxiv.org/abs/2403.10794) | 该研究提出了一种在对抗性多智能体游戏中应用扩散-强化学习的分层运动规划方法，通过整合高级扩散模型和低级RL算法，实现比基准方法更高效率的运动规划。 |
| [^2] | [Hierarchical hybrid modeling for flexible tool use](https://arxiv.org/abs/2402.10088) | 本研究基于主动推理计算框架，提出了一个分层混合模型，通过组合离散和连续模型以实现灵活工具使用，控制和规划。在非平凡任务中验证了该模型的有效性和可扩展性。 |

# 详细

[^1]: 在对抗性多智能体游戏中的扩散-强化学习分层运动规划

    Diffusion-Reinforcement Learning Hierarchical Motion Planning in Adversarial Multi-agent Games

    [https://arxiv.org/abs/2403.10794](https://arxiv.org/abs/2403.10794)

    该研究提出了一种在对抗性多智能体游戏中应用扩散-强化学习的分层运动规划方法，通过整合高级扩散模型和低级RL算法，实现比基准方法更高效率的运动规划。

    

    强化学习（RL）-基于的运动规划最近展现出优于传统方法的潜力，从自主导航到机器人操作。在这项工作中，我们关注部分可观察多智能体对抗追逐逃避游戏（PEG）中对逃避目标的运动规划任务。这些追逐逃避问题与各种应用相关，例如搜索和救援行动以及监视机器人，其中机器人必须有效规划他们的行动来收集情报或完成任务，同时避免被侦查或被俘虏自己。我们提出了一种分层架构，该架构整合了一个高级扩散模型，用于规划对环境数据敏感的全局路径，同时低级RL算法推理闪避行为与全局路径跟随行为。通过利用扩散模型引导RL算法，我们的方法比基线提高了51.2％，实现更高效率。

    arXiv:2403.10794v1 Announce Type: cross  Abstract: Reinforcement Learning- (RL-)based motion planning has recently shown the potential to outperform traditional approaches from autonomous navigation to robot manipulation. In this work, we focus on a motion planning task for an evasive target in a partially observable multi-agent adversarial pursuit-evasion games (PEG). These pursuit-evasion problems are relevant to various applications, such as search and rescue operations and surveillance robots, where robots must effectively plan their actions to gather intelligence or accomplish mission tasks while avoiding detection or capture themselves. We propose a hierarchical architecture that integrates a high-level diffusion model to plan global paths responsive to environment data while a low-level RL algorithm reasons about evasive versus global path-following behavior. Our approach outperforms baselines by 51.2% by leveraging the diffusion model to guide the RL algorithm for more efficien
    
[^2]: 分层混合建模用于灵活工具使用

    Hierarchical hybrid modeling for flexible tool use

    [https://arxiv.org/abs/2402.10088](https://arxiv.org/abs/2402.10088)

    本研究基于主动推理计算框架，提出了一个分层混合模型，通过组合离散和连续模型以实现灵活工具使用，控制和规划。在非平凡任务中验证了该模型的有效性和可扩展性。

    

    在最近提出的主动推理计算框架中，离散模型可以与连续模型相结合，以在不断变化的环境中进行决策。从另一个角度来看，简单的代理可以组合在一起，以更好地捕捉世界的因果关系。我们如何将这两个特点结合起来实现高效的目标导向行为？我们提出了一个架构，由多个混合 - 连续和离散 - 单元组成，复制代理的配置，由高级离散模型控制，实现动态规划和同步行为。每个层次内部的进一步分解可以以分层方式表示与self相关的其他代理和对象。我们在一个非平凡的任务上评估了这种分层混合模型：在拾取一个移动工具后到达一个移动物体。这项研究扩展了以推理为控制的先前工作，并提出了一种替代方案。

    arXiv:2402.10088v1 Announce Type: cross  Abstract: In a recent computational framework called active inference, discrete models can be linked to their continuous counterparts to perform decision-making in changing environments. From another perspective, simple agents can be combined to better capture the causal relationships of the world. How can we use these two features together to achieve efficient goal-directed behavior? We present an architecture composed of several hybrid -- continuous and discrete -- units replicating the agent's configuration, controlled by a high-level discrete model that achieves dynamic planning and synchronized behavior. Additional factorizations within each level allow to represent hierarchically other agents and objects in relation to the self. We evaluate this hierarchical hybrid model on a non-trivial task: reaching a moving object after having picked a moving tool. This study extends past work on control as inference and proposes an alternative directi
    

