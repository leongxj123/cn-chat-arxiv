# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Online POMDP Planning with Anytime Deterministic Guarantees.](http://arxiv.org/abs/2310.01791) | 本文中，我们推导出在线POMDP规划中一个简化解决方案与理论上最优解之间的确定性关系，以解决目前近似算法只能提供概率性和通常呈现渐进性保证的限制。 |

# 详细

[^1]: 具有任意确定性保证的在线POMDP规划

    Online POMDP Planning with Anytime Deterministic Guarantees. (arXiv:2310.01791v1 [cs.AI])

    [http://arxiv.org/abs/2310.01791](http://arxiv.org/abs/2310.01791)

    本文中，我们推导出在线POMDP规划中一个简化解决方案与理论上最优解之间的确定性关系，以解决目前近似算法只能提供概率性和通常呈现渐进性保证的限制。

    

    在现实场景中，自主智能体经常遇到不确定性并基于不完整信息做出决策。在不确定性下的规划可以使用部分可观察的马尔科夫决策过程（POMDP）进行数学建模。然而，寻找POMDP的最优规划在计算上是昂贵的，只有在小规模任务中可行。近年来，近似算法（如树搜索和基于采样的方法）已经成为解决较大问题的先进POMDP求解器。尽管这些算法有效，但它们仅提供概率性和通常呈现渐进性保证，这是由于它们依赖于采样的缘故。为了解决这些限制，我们推导出一个简化解决方案与理论上最优解之间的确定性关系。首先，我们推导出选择一组观测以在计算每个后验节点时分支的边界。

    Autonomous agents operating in real-world scenarios frequently encounter uncertainty and make decisions based on incomplete information. Planning under uncertainty can be mathematically formalized using partially observable Markov decision processes (POMDPs). However, finding an optimal plan for POMDPs can be computationally expensive and is feasible only for small tasks. In recent years, approximate algorithms, such as tree search and sample-based methodologies, have emerged as state-of-the-art POMDP solvers for larger problems. Despite their effectiveness, these algorithms offer only probabilistic and often asymptotic guarantees toward the optimal solution due to their dependence on sampling. To address these limitations, we derive a deterministic relationship between a simplified solution that is easier to obtain and the theoretically optimal one. First, we derive bounds for selecting a subset of the observations to branch from while computing a complete belief at each posterior nod
    

