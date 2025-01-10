# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Differentially Private Online Federated Learning with Correlated Noise](https://arxiv.org/abs/2403.16542) | 提出一种利用相关噪声提高效用并确保隐私的差分隐私在线联邦学习算法，解决了DP噪声和本地更新带来的挑战，并在动态环境中建立了动态遗憾界。 |

# 详细

[^1]: 具有相关噪声的差分隐私在线联邦学习

    Differentially Private Online Federated Learning with Correlated Noise

    [https://arxiv.org/abs/2403.16542](https://arxiv.org/abs/2403.16542)

    提出一种利用相关噪声提高效用并确保隐私的差分隐私在线联邦学习算法，解决了DP噪声和本地更新带来的挑战，并在动态环境中建立了动态遗憾界。

    

    我们提出了一种新颖的差分隐私算法，用于在线联邦学习，利用时间相关的噪声来提高效用同时确保连续发布的模型的隐私性。为了解决源自DP噪声和本地更新带来的流式非独立同分布数据的挑战，我们开发了扰动迭代分析来控制DP噪声对效用的影响。此外，我们展示了在准强凸条件下如何有效管理来自本地更新的漂移误差。在$(\epsilon, \delta)$-DP预算范围内，我们建立了整个时间段上的动态遗憾界，量化了关键参数的影响以及动态环境变化的强度。数值实验证实了所提算法的有效性。

    arXiv:2403.16542v1 Announce Type: new  Abstract: We propose a novel differentially private algorithm for online federated learning that employs temporally correlated noise to improve the utility while ensuring the privacy of the continuously released models. To address challenges stemming from DP noise and local updates with streaming noniid data, we develop a perturbed iterate analysis to control the impact of the DP noise on the utility. Moreover, we demonstrate how the drift errors from local updates can be effectively managed under a quasi-strong convexity condition. Subject to an $(\epsilon, \delta)$-DP budget, we establish a dynamic regret bound over the entire time horizon that quantifies the impact of key parameters and the intensity of changes in dynamic environments. Numerical experiments validate the efficacy of the proposed algorithm.
    

