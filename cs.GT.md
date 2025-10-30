# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Partially Observable Multi-agent RL with (Quasi-)Efficiency: The Blessing of Information Sharing.](http://arxiv.org/abs/2308.08705) | 本文研究了部分可观测随机博弈的可证明多Agent强化学习。通过信息共享和观测可能性假设，提出了构建近似模型以实现准效率的方法。 |

# 详细

[^1]: 部分可观测的多Agent强化学习与（准）效率：信息共享的好处。

    Partially Observable Multi-agent RL with (Quasi-)Efficiency: The Blessing of Information Sharing. (arXiv:2308.08705v1 [cs.LG])

    [http://arxiv.org/abs/2308.08705](http://arxiv.org/abs/2308.08705)

    本文研究了部分可观测随机博弈的可证明多Agent强化学习。通过信息共享和观测可能性假设，提出了构建近似模型以实现准效率的方法。

    

    本文研究了部分可观测随机博弈（POSGs）的可证明多Agent强化学习（MARL）。为了规避已知的难度问题和使用计算不可行的预言机，我们倡导利用Agent之间的潜在“信息共享”，这是实证MARL中的常见做法，也是具备通信功能的多Agent控制系统的标准模型。我们首先建立了若干计算复杂性结果，来证明信息共享的必要性，以及观测可能性假设为了求解POSGs中的计算效率已经使得部分可观测的单Agent强化学习具有准效率。然后我们提出进一步“近似”共享的公共信息构建POSG的“近似模型”，在该模型中计划一个近似均衡（从解决原始POSG的角度）可以实现准效率，即准多项式时间，前提是上述假设满足。

    We study provable multi-agent reinforcement learning (MARL) in the general framework of partially observable stochastic games (POSGs). To circumvent the known hardness results and the use of computationally intractable oracles, we advocate leveraging the potential \emph{information-sharing} among agents, a common practice in empirical MARL, and a standard model for multi-agent control systems with communications. We first establish several computation complexity results to justify the necessity of information-sharing, as well as the observability assumption that has enabled quasi-efficient single-agent RL with partial observations, for computational efficiency in solving POSGs. We then propose to further \emph{approximate} the shared common information to construct an {approximate model} of the POSG, in which planning an approximate equilibrium (in terms of solving the original POSG) can be quasi-efficient, i.e., of quasi-polynomial-time, under the aforementioned assumptions. Furthermo
    

