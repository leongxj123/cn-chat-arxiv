# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GradSkip: Communication-Accelerated Local Gradient Methods with Better Computational Complexity.](http://arxiv.org/abs/2210.16402) | 本文研究了一类分布式优化算法，通过允许具有“次要”数据的客户端在本地执行较少的训练步骤来减轻高通信成本，这一方法可在强凸区域内实现可证明的通信加速。 |

# 详细

[^1]: GradSkip：具有更好计算复杂度的通信加速局部梯度方法

    GradSkip: Communication-Accelerated Local Gradient Methods with Better Computational Complexity. (arXiv:2210.16402v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.16402](http://arxiv.org/abs/2210.16402)

    本文研究了一类分布式优化算法，通过允许具有“次要”数据的客户端在本地执行较少的训练步骤来减轻高通信成本，这一方法可在强凸区域内实现可证明的通信加速。

    

    我们研究了一类分布式优化算法，旨在通过允许客户端在通信之前执行多个本地梯度类型的训练步骤来减轻高通信成本。虽然这种方法已经研究了约十年，但本地训练的加速性质在理论上还未得到完全解释。最近，Mishchenko等人(2022 International Conference on Machine Learning)取得了重大突破，证明了当本地训练得到正确执行时，会导致可证明的通信加速，在强凸区域内这一点成立，而且不依赖于任何数据相似性假设。然而，他们的方法ProxSkip要求所有客户端在每次通信轮中执行相同数量的本地训练步骤。灵感来自常识的直觉，我们通过猜测认为拥有“次要”数据的客户端应该能够用较少的本地训练步骤就能完成，而不影响整体通信

    We study a class of distributed optimization algorithms that aim to alleviate high communication costs by allowing the clients to perform multiple local gradient-type training steps prior to communication. While methods of this type have been studied for about a decade, the empirically observed acceleration properties of local training eluded all attempts at theoretical understanding. In a recent breakthrough, Mishchenko et al. (ICML 2022) proved that local training, when properly executed, leads to provable communication acceleration, and this holds in the strongly convex regime without relying on any data similarity assumptions. However, their method ProxSkip requires all clients to take the same number of local training steps in each communication round. Inspired by a common sense intuition, we start our investigation by conjecturing that clients with ``less important'' data should be able to get away with fewer local training steps without this impacting the overall communication c
    

