# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sequence graphs realizations and ambiguity in language models](https://arxiv.org/abs/2402.08830) | 本文研究了语言模型中序列图的实现和歧义问题，通过组合和计算的方法考虑了图的窗口大小、方向性和权重等因素，并提供了多项式时间算法来解决实现和枚举问题。 |

# 详细

[^1]: 序列图实现与语言模型中的歧义

    Sequence graphs realizations and ambiguity in language models

    [https://arxiv.org/abs/2402.08830](https://arxiv.org/abs/2402.08830)

    本文研究了语言模型中序列图的实现和歧义问题，通过组合和计算的方法考虑了图的窗口大小、方向性和权重等因素，并提供了多项式时间算法来解决实现和枚举问题。

    

    几种流行的语言模型将输入文本中的局部上下文表示为词袋。这样的表示自然地通过一个序列图来编码，其中顶点是出现在输入文本中的不同词，边表示在大小为w的滑动窗口内两个词的（有序）共现。然而，这种压缩表示通常不是双射的，可能引入一定程度的歧义。一些序列图可能以多种方式实现为一个序列，而其他一些可能无法实现任何序列。在本文中，我们从组合和计算的角度研究了序列图的可实现性和歧义。我们考虑在多种设置下的序列图实现的存在和枚举：窗口大小w、图的方向性的存在/缺失和权重（重复性）的存在/缺失。当w = 2时，我们提供了多项式时间算法来实现和枚举。

    arXiv:2402.08830v1 Announce Type: cross Abstract: Several popular language models represent local contexts in an input text as bags of words. Such representations are naturally encoded by a sequence graph whose vertices are the distinct words occurring in x, with edges representing the (ordered) co-occurrence of two words within a sliding window of size w. However, this compressed representation is not generally bijective, and may introduce some degree of ambiguity. Some sequence graphs may admit several realizations as a sequence, while others may not admit any realization. In this paper, we study the realizability and ambiguity of sequence graphs from a combinatorial and computational point of view. We consider the existence and enumeration of realizations of a sequence graph under multiple settings: window size w, presence/absence of graph orientation, and presence/absence of weights (multiplicities). When w = 2, we provide polynomial time algorithms for realizability and enumeratio
    

