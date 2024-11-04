# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reinforced In-Context Black-Box Optimization](https://arxiv.org/abs/2402.17423) | 提出了一种从离线数据中端到端地强化学习黑盒优化算法的方法，通过使用表达能力强的序列模型和后悔-前进令牌来获取任务信息并做出决策。 |

# 详细

[^1]: 加强上下文黑盒优化

    Reinforced In-Context Black-Box Optimization

    [https://arxiv.org/abs/2402.17423](https://arxiv.org/abs/2402.17423)

    提出了一种从离线数据中端到端地强化学习黑盒优化算法的方法，通过使用表达能力强的序列模型和后悔-前进令牌来获取任务信息并做出决策。

    

    黑盒优化（BBO）已经在许多科学和工程领域取得成功应用。最近，人们越来越关注元学习BBO算法的特定组件，以加快优化速度并摆脱繁琐的手工启发式算法。作为扩展，从数据中学习整个算法需要专家最少的工作量，并且可以提供最大的灵活性。在本文中，我们提出了一种名为RIBBO的方法，可以以端到端的方式从离线数据中强化学习BBO算法。RIBBO利用表达能力强的序列模型来学习多个行为算法和任务产生的优化历史，利用大型模型的上下文学习能力来提取任务信息并相应地做出决策。我们方法的核心是通过增加后悔-前进令牌来增强优化历史，这些令牌旨在基于累积表现来表示算法的性能。

    arXiv:2402.17423v1 Announce Type: cross  Abstract: Black-Box Optimization (BBO) has found successful applications in many fields of science and engineering. Recently, there has been a growing interest in meta-learning particular components of BBO algorithms to speed up optimization and get rid of tedious hand-crafted heuristics. As an extension, learning the entire algorithm from data requires the least labor from experts and can provide the most flexibility. In this paper, we propose RIBBO, a method to reinforce-learn a BBO algorithm from offline data in an end-to-end fashion. RIBBO employs expressive sequence models to learn the optimization histories produced by multiple behavior algorithms and tasks, leveraging the in-context learning ability of large models to extract task information and make decisions accordingly. Central to our method is to augment the optimization histories with regret-to-go tokens, which are designed to represent the performance of an algorithm based on cumul
    

