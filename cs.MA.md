# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diffusion-Reinforcement Learning Hierarchical Motion Planning in Adversarial Multi-agent Games](https://arxiv.org/abs/2403.10794) | 该研究提出了一种在对抗性多智能体游戏中应用扩散-强化学习的分层运动规划方法，通过整合高级扩散模型和低级RL算法，实现比基准方法更高效率的运动规划。 |

# 详细

[^1]: 在对抗性多智能体游戏中的扩散-强化学习分层运动规划

    Diffusion-Reinforcement Learning Hierarchical Motion Planning in Adversarial Multi-agent Games

    [https://arxiv.org/abs/2403.10794](https://arxiv.org/abs/2403.10794)

    该研究提出了一种在对抗性多智能体游戏中应用扩散-强化学习的分层运动规划方法，通过整合高级扩散模型和低级RL算法，实现比基准方法更高效率的运动规划。

    

    强化学习（RL）-基于的运动规划最近展现出优于传统方法的潜力，从自主导航到机器人操作。在这项工作中，我们关注部分可观察多智能体对抗追逐逃避游戏（PEG）中对逃避目标的运动规划任务。这些追逐逃避问题与各种应用相关，例如搜索和救援行动以及监视机器人，其中机器人必须有效规划他们的行动来收集情报或完成任务，同时避免被侦查或被俘虏自己。我们提出了一种分层架构，该架构整合了一个高级扩散模型，用于规划对环境数据敏感的全局路径，同时低级RL算法推理闪避行为与全局路径跟随行为。通过利用扩散模型引导RL算法，我们的方法比基线提高了51.2％，实现更高效率。

    arXiv:2403.10794v1 Announce Type: cross  Abstract: Reinforcement Learning- (RL-)based motion planning has recently shown the potential to outperform traditional approaches from autonomous navigation to robot manipulation. In this work, we focus on a motion planning task for an evasive target in a partially observable multi-agent adversarial pursuit-evasion games (PEG). These pursuit-evasion problems are relevant to various applications, such as search and rescue operations and surveillance robots, where robots must effectively plan their actions to gather intelligence or accomplish mission tasks while avoiding detection or capture themselves. We propose a hierarchical architecture that integrates a high-level diffusion model to plan global paths responsive to environment data while a low-level RL algorithm reasons about evasive versus global path-following behavior. Our approach outperforms baselines by 51.2% by leveraging the diffusion model to guide the RL algorithm for more efficien
    

