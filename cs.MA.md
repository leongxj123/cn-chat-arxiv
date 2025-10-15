# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Offline Fictitious Self-Play for Competitive Games](https://arxiv.org/abs/2403.00841) | 本文介绍了Off-FSP，这是竞争游戏的第一个实用的无模型离线RL算法，通过调整固定数据集的权重，使用重要性抽样，模拟与各种对手的互动。 |
| [^2] | [Optimistic Multi-Agent Policy Gradient for Cooperative Tasks.](http://arxiv.org/abs/2311.01953) | 本论文提出了一个通用的、简单的框架，在多智能体策略梯度方法中引入乐观更新，以缓解合作任务中的相对过度概括问题。通过使用一个泄漏化线性整流函数来重塑优势，我们的方法能够保持对潜在由其他代理引起的低回报个别动作的乐观态度。 |

# 详细

[^1]: 竞争游戏的离线虚构自我对弈

    Offline Fictitious Self-Play for Competitive Games

    [https://arxiv.org/abs/2403.00841](https://arxiv.org/abs/2403.00841)

    本文介绍了Off-FSP，这是竞争游戏的第一个实用的无模型离线RL算法，通过调整固定数据集的权重，使用重要性抽样，模拟与各种对手的互动。

    

    离线强化学习（RL）因其在以前收集的数据集中改进策略而不需要在线交互的能力而受到重视。尽管在单一智能体设置中取得成功，但离线多智能体RL仍然是一个挑战，特别是在竞争游戏中。为了解决这些问题，本文介绍了Off-FSP，这是竞争游戏的第一个实用的无模型离线RL算法。我们首先通过调整固定数据集的权重，使用重要性抽样模拟与各种对手的互动。

    arXiv:2403.00841v1 Announce Type: cross  Abstract: Offline Reinforcement Learning (RL) has received significant interest due to its ability to improve policies in previously collected datasets without online interactions. Despite its success in the single-agent setting, offline multi-agent RL remains a challenge, especially in competitive games. Firstly, unaware of the game structure, it is impossible to interact with the opponents and conduct a major learning paradigm, self-play, for competitive games. Secondly, real-world datasets cannot cover all the state and action space in the game, resulting in barriers to identifying Nash equilibrium (NE). To address these issues, this paper introduces Off-FSP, the first practical model-free offline RL algorithm for competitive games. We start by simulating interactions with various opponents by adjusting the weights of the fixed dataset with importance sampling. This technique allows us to learn best responses to different opponents and employ
    
[^2]: 乐观的多智能体策略梯度在合作任务中的应用

    Optimistic Multi-Agent Policy Gradient for Cooperative Tasks. (arXiv:2311.01953v1 [cs.LG])

    [http://arxiv.org/abs/2311.01953](http://arxiv.org/abs/2311.01953)

    本论文提出了一个通用的、简单的框架，在多智能体策略梯度方法中引入乐观更新，以缓解合作任务中的相对过度概括问题。通过使用一个泄漏化线性整流函数来重塑优势，我们的方法能够保持对潜在由其他代理引起的低回报个别动作的乐观态度。

    

    在合作多智能体学习任务中，由于过拟合其他智能体的次优行为，导致智能体收敛到次优联合策略，出现了相对过度概括（RO）问题。早期研究表明，乐观主义可以缓解使用表格化Q学习时的RO问题。然而，对于复杂任务来说，利用函数逼近乐观主义可能加剧过估计，从而失败。另一方面，最近的深度多智能体策略梯度（MAPG）方法在许多复杂任务上取得了成功，但在严重的RO情况下可能失败。我们提出了一个通用而简单的框架，以在MAPG方法中实现乐观更新并缓解RO问题。具体而言，我们使用一个泄漏化线性整流函数，其中一个超参数选择乐观程度以在更新策略时重新塑造优势。直观地说，我们的方法对可能由其他代理引起的回报较低的个别动作保持乐观态度。

    \textit{Relative overgeneralization} (RO) occurs in cooperative multi-agent learning tasks when agents converge towards a suboptimal joint policy due to overfitting to suboptimal behavior of other agents. In early work, optimism has been shown to mitigate the \textit{RO} problem when using tabular Q-learning. However, with function approximation optimism can amplify overestimation and thus fail on complex tasks. On the other hand, recent deep multi-agent policy gradient (MAPG) methods have succeeded in many complex tasks but may fail with severe \textit{RO}. We propose a general, yet simple, framework to enable optimistic updates in MAPG methods and alleviate the RO problem. Specifically, we employ a \textit{Leaky ReLU} function where a single hyperparameter selects the degree of optimism to reshape the advantages when updating the policy. Intuitively, our method remains optimistic toward individual actions with lower returns which are potentially caused by other agents' sub-optimal be
    

