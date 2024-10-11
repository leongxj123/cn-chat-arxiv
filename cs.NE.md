# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Positional Encoding Helps Recurrent Neural Networks Handle a Large Vocabulary](https://arxiv.org/abs/2402.00236) | 本研究探讨了位置编码在循环神经网络中的作用，发现即使与RNN结合使用，位置编码仍然有效，尤其是在处理大词汇量和多样观察结果时。这为使用输入驱动和自主时间表示的组合研究提供了新的方向，同时研究结果也对神经元振荡的生物学意义提供了讨论。 |
| [^2] | [Multi-Modal Optimization with k-Cluster Big Bang-Big Crunch Algorithm.](http://arxiv.org/abs/2401.06153) | 这篇论文提出了一种基于聚类的多模态优化Big Bang-Big Crunch算法的版本，称为k-BBBC。该算法能够高效地解决多模态优化问题，并在测试中表现出良好的性能。 |

# 详细

[^1]: 位置编码有助于循环神经网络处理大词汇量

    Positional Encoding Helps Recurrent Neural Networks Handle a Large Vocabulary

    [https://arxiv.org/abs/2402.00236](https://arxiv.org/abs/2402.00236)

    本研究探讨了位置编码在循环神经网络中的作用，发现即使与RNN结合使用，位置编码仍然有效，尤其是在处理大词汇量和多样观察结果时。这为使用输入驱动和自主时间表示的组合研究提供了新的方向，同时研究结果也对神经元振荡的生物学意义提供了讨论。

    

    本研究讨论了位置编码在利用合成基准测试的循环神经网络（RNN）中的影响。位置编码将时间序列中的数据点“时间戳化”，并补充了Transformer神经网络的能力，后者缺乏表示数据顺序的内在机制。相反，RNN可以自己对数据点进行时间编码，使得它们对位置编码的使用似乎是“冗余”的。然而，经验研究表明，即使与RNN结合使用，位置编码的有效性仍然很高，特别是用于处理产生多样观察结果的大词汇量。这些发现为循环神经网络上的新的研究方向铺平了道路，涉及输入驱动和自主时间表示的组合。此外，本研究还讨论了计算/模拟结果的生物学意义，考虑到位置编码的正弦实现与神经元振荡之间的关联。

    This study discusses the effects of positional encoding on recurrent neural networks (RNNs) utilizing synthetic benchmarks. Positional encoding "time-stamps" data points in time series and complements the capabilities of Transformer neural networks, which lack an inherent mechanism for representing the data order. By contrast, RNNs can encode the temporal information of data points on their own, rendering their use of positional encoding seemingly "redundant". Nonetheless, empirical investigations reveal the effectiveness of positional encoding even when coupled with RNNs, specifically for handling a large vocabulary that yields diverse observations. These findings pave the way for a new line of research on RNNs, concerning the combination of input-driven and autonomous time representation. Additionally, biological implications of the computational/simulational results are discussed, in the light of the affinity between the sinusoidal implementation of positional encoding and neural os
    
[^2]: 基于k-聚类Big Bang-Big Crunch算法的多模态优化

    Multi-Modal Optimization with k-Cluster Big Bang-Big Crunch Algorithm. (arXiv:2401.06153v1 [cs.NE])

    [http://arxiv.org/abs/2401.06153](http://arxiv.org/abs/2401.06153)

    这篇论文提出了一种基于聚类的多模态优化Big Bang-Big Crunch算法的版本，称为k-BBBC。该算法能够高效地解决多模态优化问题，并在测试中表现出良好的性能。

    

    多模态优化经常在工程问题中遇到，特别是在寻找不同和替代解决方案时。进化算法通过种群的概念、探索/开发功能和适合并行计算等特点，能够高效地解决多模态优化问题。本文介绍了一种基于聚类的多模态优化Big Bang-Big Crunch算法的版本，称为k-BBBC。该算法能够保证整个种群的完全收敛，对于特定问题平均检索到99\%的局部最优解。此外，我们引入了两种后处理方法，用于(i)在一组检索到的解决方案中确定局部最优解，以及(ii)定量测量正确检索到的最优解数量与预期数量之间的比率（即成功率）。我们的结果表明，k-BBBC在具有大量最优解（测试了379个最优解）和高维度的问题上表现良好。

    Multi-modal optimization is often encountered in engineering problems, especially when different and alternative solutions are sought. Evolutionary algorithms can efficiently tackle multi-modal optimization thanks to their features such as the concept of population, exploration/exploitation, and being suitable for parallel computation.  This paper introduces a multi-modal optimization version of the Big Bang-Big Crunch algorithm based on clustering, namely, k-BBBC. This algorithm guarantees a complete convergence of the entire population, retrieving on average the 99\% of local optima for a specific problem. Additionally, we introduce two post-processing methods to (i) identify the local optima in a set of retrieved solutions (i.e., a population), and (ii) quantify the number of correctly retrieved optima against the expected ones (i.e., success rate).  Our results show that k-BBBC performs well even with problems having a large number of optima (tested on 379 optima) and high dimensio
    

