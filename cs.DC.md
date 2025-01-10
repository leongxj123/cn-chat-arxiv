# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Differentially Private Online Federated Learning with Correlated Noise](https://arxiv.org/abs/2403.16542) | 提出一种利用相关噪声提高效用并确保隐私的差分隐私在线联邦学习算法，解决了DP噪声和本地更新带来的挑战，并在动态环境中建立了动态遗憾界。 |
| [^2] | [Convergence Analysis of Split Federated Learning on Heterogeneous Data](https://arxiv.org/abs/2402.15166) | 本文填补了分裂联邦学习在各异数据上收敛分析的空白，提供了针对强凸和一般凸目标的SFL收敛分析，收敛速率分别为$O(1/T)$和$O(1/\sqrt[3]{T})。 |

# 详细

[^1]: 具有相关噪声的差分隐私在线联邦学习

    Differentially Private Online Federated Learning with Correlated Noise

    [https://arxiv.org/abs/2403.16542](https://arxiv.org/abs/2403.16542)

    提出一种利用相关噪声提高效用并确保隐私的差分隐私在线联邦学习算法，解决了DP噪声和本地更新带来的挑战，并在动态环境中建立了动态遗憾界。

    

    我们提出了一种新颖的差分隐私算法，用于在线联邦学习，利用时间相关的噪声来提高效用同时确保连续发布的模型的隐私性。为了解决源自DP噪声和本地更新带来的流式非独立同分布数据的挑战，我们开发了扰动迭代分析来控制DP噪声对效用的影响。此外，我们展示了在准强凸条件下如何有效管理来自本地更新的漂移误差。在$(\epsilon, \delta)$-DP预算范围内，我们建立了整个时间段上的动态遗憾界，量化了关键参数的影响以及动态环境变化的强度。数值实验证实了所提算法的有效性。

    arXiv:2403.16542v1 Announce Type: new  Abstract: We propose a novel differentially private algorithm for online federated learning that employs temporally correlated noise to improve the utility while ensuring the privacy of the continuously released models. To address challenges stemming from DP noise and local updates with streaming noniid data, we develop a perturbed iterate analysis to control the impact of the DP noise on the utility. Moreover, we demonstrate how the drift errors from local updates can be effectively managed under a quasi-strong convexity condition. Subject to an $(\epsilon, \delta)$-DP budget, we establish a dynamic regret bound over the entire time horizon that quantifies the impact of key parameters and the intensity of changes in dynamic environments. Numerical experiments validate the efficacy of the proposed algorithm.
    
[^2]: 分布式异构数据上的分裂联邦学习的收敛分析

    Convergence Analysis of Split Federated Learning on Heterogeneous Data

    [https://arxiv.org/abs/2402.15166](https://arxiv.org/abs/2402.15166)

    本文填补了分裂联邦学习在各异数据上收敛分析的空白，提供了针对强凸和一般凸目标的SFL收敛分析，收敛速率分别为$O(1/T)$和$O(1/\sqrt[3]{T})。

    

    分裂联邦学习（SFL）是一种最近的分布式方法，用于在多个客户端之间进行协作模型训练。在SFL中，全局模型通常被分为两部分，其中客户端以并行联邦方式训练一部分，主服务器训练另一部分。尽管最近关于SFL算法发展的研究很多，但SFL的收敛分析在文献中还未有提及，本文旨在弥补这一空白。对SFL进行分析可能比对联邦学习（FL）的分析更具挑战性，这是由于客户端和主服务器之间可能存在双速更新。我们提供了针对异构数据上强凸和一般凸目标的SFL收敛分析。收敛速率分别为$O(1/T)$和$O(1/\sqrt[3]{T})$，其中$T$表示SFL训练的总轮数。我们进一步将分析扩展到非凸目标和一些客户端可能在训练过程中不可用的情况。

    arXiv:2402.15166v1 Announce Type: cross  Abstract: Split federated learning (SFL) is a recent distributed approach for collaborative model training among multiple clients. In SFL, a global model is typically split into two parts, where clients train one part in a parallel federated manner, and a main server trains the other. Despite the recent research on SFL algorithm development, the convergence analysis of SFL is missing in the literature, and this paper aims to fill this gap. The analysis of SFL can be more challenging than that of federated learning (FL), due to the potential dual-paced updates at the clients and the main server. We provide convergence analysis of SFL for strongly convex and general convex objectives on heterogeneous data. The convergence rates are $O(1/T)$ and $O(1/\sqrt[3]{T})$, respectively, where $T$ denotes the total number of rounds for SFL training. We further extend the analysis to non-convex objectives and where some clients may be unavailable during trai
    

