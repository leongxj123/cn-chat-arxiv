# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Online Local False Discovery Rate Control: A Resource Allocation Approach](https://arxiv.org/abs/2402.11425) | 该研究提出了一种在线局部虚发现率控制的资源分配方法，实现了$O(\sqrt{T})$的后悔率，并指出这种后悔率在一般情况下是不可改进的。 |

# 详细

[^1]: 在线局部虚发现率控制：一种资源分配方法

    Online Local False Discovery Rate Control: A Resource Allocation Approach

    [https://arxiv.org/abs/2402.11425](https://arxiv.org/abs/2402.11425)

    该研究提出了一种在线局部虚发现率控制的资源分配方法，实现了$O(\sqrt{T})$的后悔率，并指出这种后悔率在一般情况下是不可改进的。

    

    我们考虑在线局部虚发现率（FDR）控制问题，其中多个测试被顺序进行，目标是最大化总期望的发现次数。我们将问题形式化为一种在线资源分配问题，涉及接受/拒绝决策，从高层次来看，这可以被视为一个带有额外不确定性的在线背包问题，即随机预算补充。我们从一般的到达分布开始，并提出了一个简单的策略，实现了$O(\sqrt{T})$的后悔。我们通过展示这种后悔率在一般情况下是不可改进的来补充这一结果。然后我们将焦点转向离散到达分布。我们发现许多现有的在线资源分配文献中的重新解决启发式虽然在典型设置中实现了有界的损失，但可能会造成$\Omega(\sqrt{T})$甚至$\Omega(T)$的后悔。通过观察到典型策略往往太过

    arXiv:2402.11425v1 Announce Type: cross  Abstract: We consider the problem of online local false discovery rate (FDR) control where multiple tests are conducted sequentially, with the goal of maximizing the total expected number of discoveries. We formulate the problem as an online resource allocation problem with accept/reject decisions, which from a high level can be viewed as an online knapsack problem, with the additional uncertainty of random budget replenishment. We start with general arrival distributions and propose a simple policy that achieves a $O(\sqrt{T})$ regret. We complement the result by showing that such regret rate is in general not improvable. We then shift our focus to discrete arrival distributions. We find that many existing re-solving heuristics in the online resource allocation literature, albeit achieve bounded loss in canonical settings, may incur a $\Omega(\sqrt{T})$ or even a $\Omega(T)$ regret. With the observation that canonical policies tend to be too op
    

