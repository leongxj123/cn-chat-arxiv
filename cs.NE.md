# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reinforced In-Context Black-Box Optimization](https://arxiv.org/abs/2402.17423) | 提出了一种从离线数据中端到端地强化学习黑盒优化算法的方法，通过使用表达能力强的序列模型和后悔-前进令牌来获取任务信息并做出决策。 |
| [^2] | [What's the Magic Word? A Control Theory of LLM Prompting.](http://arxiv.org/abs/2310.04444) | 本论文将提示工程形式化为LLM上的最优控制问题，研究了给定token序列时是否存在一种最优提示能够准确预测最终的token，并提出了控制理论中的指标来描述LLM的可操纵性。 |

# 详细

[^1]: 加强上下文黑盒优化

    Reinforced In-Context Black-Box Optimization

    [https://arxiv.org/abs/2402.17423](https://arxiv.org/abs/2402.17423)

    提出了一种从离线数据中端到端地强化学习黑盒优化算法的方法，通过使用表达能力强的序列模型和后悔-前进令牌来获取任务信息并做出决策。

    

    黑盒优化（BBO）已经在许多科学和工程领域取得成功应用。最近，人们越来越关注元学习BBO算法的特定组件，以加快优化速度并摆脱繁琐的手工启发式算法。作为扩展，从数据中学习整个算法需要专家最少的工作量，并且可以提供最大的灵活性。在本文中，我们提出了一种名为RIBBO的方法，可以以端到端的方式从离线数据中强化学习BBO算法。RIBBO利用表达能力强的序列模型来学习多个行为算法和任务产生的优化历史，利用大型模型的上下文学习能力来提取任务信息并相应地做出决策。我们方法的核心是通过增加后悔-前进令牌来增强优化历史，这些令牌旨在基于累积表现来表示算法的性能。

    arXiv:2402.17423v1 Announce Type: cross  Abstract: Black-Box Optimization (BBO) has found successful applications in many fields of science and engineering. Recently, there has been a growing interest in meta-learning particular components of BBO algorithms to speed up optimization and get rid of tedious hand-crafted heuristics. As an extension, learning the entire algorithm from data requires the least labor from experts and can provide the most flexibility. In this paper, we propose RIBBO, a method to reinforce-learn a BBO algorithm from offline data in an end-to-end fashion. RIBBO employs expressive sequence models to learn the optimization histories produced by multiple behavior algorithms and tasks, leveraging the in-context learning ability of large models to extract task information and make decisions accordingly. Central to our method is to augment the optimization histories with regret-to-go tokens, which are designed to represent the performance of an algorithm based on cumul
    
[^2]: 魔法词是什么？LLM提示的控制理论研究

    What's the Magic Word? A Control Theory of LLM Prompting. (arXiv:2310.04444v1 [cs.CL])

    [http://arxiv.org/abs/2310.04444](http://arxiv.org/abs/2310.04444)

    本论文将提示工程形式化为LLM上的最优控制问题，研究了给定token序列时是否存在一种最优提示能够准确预测最终的token，并提出了控制理论中的指标来描述LLM的可操纵性。

    

    提示工程在LLM的部署中是有效和重要的，但在数学上理解不足。在这里，我们将提示工程形式化为LLM上的最优控制问题，其中提示被认为是调节LLM输出分布的控制变量。在这个框架内，我们提出一个简单的问题：给定一个token序列，是否总存在一个我们可以添加的提示，使得LLM能够准确预测最终的token？我们将这样的最优提示称为魔法词，因为添加提示会导致LLM输出正确的答案。如果存在魔法词，我们能否找到它们？如果可以，它们的特性是什么？我们提供了将控制理论应用于自注意力头的分析分析，证明了其权重矩阵的奇异值函数为可控制性的上界。我们借鉴控制理论来提出了一种叫做$k-\epsilon$可控制性的指标，用于描述LLM的可操纵性。

    Prompt engineering is effective and important in the deployment of LLMs but is poorly understood mathematically. Here, we formalize prompt engineering as an optimal control problem on LLMs -- where the prompt is considered a control variable for modulating the output distribution of the LLM. Within this framework, we ask a simple question: given a sequence of tokens, does there always exist a prompt we can prepend that will steer the LLM toward accurately predicting the final token? We call such an optimal prompt the magic word since prepending the prompt causes the LLM to output the correct answer. If magic words exist, can we find them? If so, what are their properties? We offer analytic analysis on the controllability of the self-attention head where we prove a bound on controllability as a function of the singular values of its weight matrices. We take inspiration from control theory to propose a metric called $k-\epsilon$ controllability to characterize LLM steerability. We comput
    

