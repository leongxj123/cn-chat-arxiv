# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Online Control of Linear Systems with Unbounded and Degenerate Noise](https://arxiv.org/abs/2402.10252) | 这项研究揭示了在线控制问题中，对于凸成本，可以实现 $ \widetilde{O}(\sqrt{T}) $ 的遗憾界，甚至在存在无界噪声的情况下；同时，在成本具有强凸性时，可以在不需要噪声协方差是非退化的情况下建立 $ O({\rm poly} (\log T)) $ 的遗憾界。 |

# 详细

[^1]: 具有无界和退化噪声的线性系统在线控制

    Online Control of Linear Systems with Unbounded and Degenerate Noise

    [https://arxiv.org/abs/2402.10252](https://arxiv.org/abs/2402.10252)

    这项研究揭示了在线控制问题中，对于凸成本，可以实现 $ \widetilde{O}(\sqrt{T}) $ 的遗憾界，甚至在存在无界噪声的情况下；同时，在成本具有强凸性时，可以在不需要噪声协方差是非退化的情况下建立 $ O({\rm poly} (\log T)) $ 的遗憾界。

    

    本文研究了在可能存在无界和退化噪声的情况下控制线性系统的问题，其中成本函数未知，被称为在线控制问题。与现有的仅假设噪声有界性的研究不同，我们揭示了对于凸成本，即使在存在无界噪声的情况下也可以实现 $ \widetilde{O}(\sqrt{T}) $ 的遗憾界，其中 $ T $ 表示时间跨度。此外，当成本具有强凸性时，我们建立了一个 $ O({\rm poly} (\log T)) $ 的遗憾界，而不需要噪声协方差是非退化的假设，这在文献中是必需的。消除噪声秩的关键是与噪声协方差相关联的系统转化。这同时实现了在线控制算法的参数减少。

    arXiv:2402.10252v1 Announce Type: cross  Abstract: This paper investigates the problem of controlling a linear system under possibly unbounded and degenerate noise with unknown cost functions, known as an online control problem. In contrast to the existing work, which assumes the boundedness of noise, we reveal that for convex costs, an $ \widetilde{O}(\sqrt{T}) $ regret bound can be achieved even for unbounded noise, where $ T $ denotes the time horizon. Moreover, when the costs are strongly convex, we establish an $ O({\rm poly} (\log T)) $ regret bound without the assumption that noise covariance is non-degenerate, which has been required in the literature. The key ingredient in removing the rank assumption on noise is a system transformation associated with the noise covariance. This simultaneously enables the parameter reduction of an online control algorithm.
    

