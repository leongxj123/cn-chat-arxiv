# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Positional Encoding Helps Recurrent Neural Networks Handle a Large Vocabulary](https://arxiv.org/abs/2402.00236) | 本研究探讨了位置编码在循环神经网络中的作用，发现即使与RNN结合使用，位置编码仍然有效，尤其是在处理大词汇量和多样观察结果时。这为使用输入驱动和自主时间表示的组合研究提供了新的方向，同时研究结果也对神经元振荡的生物学意义提供了讨论。 |

# 详细

[^1]: 位置编码有助于循环神经网络处理大词汇量

    Positional Encoding Helps Recurrent Neural Networks Handle a Large Vocabulary

    [https://arxiv.org/abs/2402.00236](https://arxiv.org/abs/2402.00236)

    本研究探讨了位置编码在循环神经网络中的作用，发现即使与RNN结合使用，位置编码仍然有效，尤其是在处理大词汇量和多样观察结果时。这为使用输入驱动和自主时间表示的组合研究提供了新的方向，同时研究结果也对神经元振荡的生物学意义提供了讨论。

    

    本研究讨论了位置编码在利用合成基准测试的循环神经网络（RNN）中的影响。位置编码将时间序列中的数据点“时间戳化”，并补充了Transformer神经网络的能力，后者缺乏表示数据顺序的内在机制。相反，RNN可以自己对数据点进行时间编码，使得它们对位置编码的使用似乎是“冗余”的。然而，经验研究表明，即使与RNN结合使用，位置编码的有效性仍然很高，特别是用于处理产生多样观察结果的大词汇量。这些发现为循环神经网络上的新的研究方向铺平了道路，涉及输入驱动和自主时间表示的组合。此外，本研究还讨论了计算/模拟结果的生物学意义，考虑到位置编码的正弦实现与神经元振荡之间的关联。

    This study discusses the effects of positional encoding on recurrent neural networks (RNNs) utilizing synthetic benchmarks. Positional encoding "time-stamps" data points in time series and complements the capabilities of Transformer neural networks, which lack an inherent mechanism for representing the data order. By contrast, RNNs can encode the temporal information of data points on their own, rendering their use of positional encoding seemingly "redundant". Nonetheless, empirical investigations reveal the effectiveness of positional encoding even when coupled with RNNs, specifically for handling a large vocabulary that yields diverse observations. These findings pave the way for a new line of research on RNNs, concerning the combination of input-driven and autonomous time representation. Additionally, biological implications of the computational/simulational results are discussed, in the light of the affinity between the sinusoidal implementation of positional encoding and neural os
    

