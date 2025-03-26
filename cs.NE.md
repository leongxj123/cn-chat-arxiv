# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Polysemanticity and Capacity in Neural Networks.](http://arxiv.org/abs/2210.01892) | 该论文通过分析特征容量来理解神经网络中的多义性现象，发现在最优的容量分配下，神经网络倾向于单义地表示重要特征，多义地表示次重要特征，并忽略最不重要的特征。多义性现象在输入具有更高的峰度或稀疏性时更为普遍，并且在某些体系结构中比其他体系结构更为普遍。此外，作者还发现了嵌入空间中的分块半正交结构，不同模型中的分块大小不同，突出了模型体系结构的影响。 |

# 详细

[^1]: 神经网络中的多义性和容量

    Polysemanticity and Capacity in Neural Networks. (arXiv:2210.01892v3 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2210.01892](http://arxiv.org/abs/2210.01892)

    该论文通过分析特征容量来理解神经网络中的多义性现象，发现在最优的容量分配下，神经网络倾向于单义地表示重要特征，多义地表示次重要特征，并忽略最不重要的特征。多义性现象在输入具有更高的峰度或稀疏性时更为普遍，并且在某些体系结构中比其他体系结构更为普遍。此外，作者还发现了嵌入空间中的分块半正交结构，不同模型中的分块大小不同，突出了模型体系结构的影响。

    

    神经网络中的单个神经元通常代表无关特征的混合。这种现象称为多义性，使解释神经网络变得更加困难，因此我们的目标是理解其原因。我们提出通过特征容量的视角来理解这一现象，特征容量是每个特征在嵌入空间中占用的分形维度。我们展示了在一个玩具模型中，最优的容量分配倾向于单义地表示最重要的特征，多义地表示次重要特征（与其对损失的影响成比例），并完全忽略最不重要的特征。当输入具有更高的峰度或稀疏性时，多义性更为普遍，并且在某些体系结构中比其他体系结构更为普遍。在得到最优容量分配后，我们进一步研究了嵌入空间的几何结构。我们发现了一个分块半正交的结构，不同模型中的分块大小不同，突出了模型体系结构的影响。

    Individual neurons in neural networks often represent a mixture of unrelated features. This phenomenon, called polysemanticity, can make interpreting neural networks more difficult and so we aim to understand its causes. We propose doing so through the lens of feature \emph{capacity}, which is the fractional dimension each feature consumes in the embedding space. We show that in a toy model the optimal capacity allocation tends to monosemantically represent the most important features, polysemantically represent less important features (in proportion to their impact on the loss), and entirely ignore the least important features. Polysemanticity is more prevalent when the inputs have higher kurtosis or sparsity and more prevalent in some architectures than others. Given an optimal allocation of capacity, we go on to study the geometry of the embedding space. We find a block-semi-orthogonal structure, with differing block sizes in different models, highlighting the impact of model archit
    

