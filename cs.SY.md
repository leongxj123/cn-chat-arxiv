# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Attention-Based Transformer Networks for Quantum State Tomography.](http://arxiv.org/abs/2305.05433) | 本研究提出一种基于注意力机制和变压器网络的 QST 方法，可捕捉不同测量之间的相关性，并成功应用于检索量子态的密度矩阵，特别是对于受限测量数据的情况表现良好。 |

# 详细

[^1]: 基于注意力机制的变压器网络用于量子态重构

    Attention-Based Transformer Networks for Quantum State Tomography. (arXiv:2305.05433v1 [quant-ph])

    [http://arxiv.org/abs/2305.05433](http://arxiv.org/abs/2305.05433)

    本研究提出一种基于注意力机制和变压器网络的 QST 方法，可捕捉不同测量之间的相关性，并成功应用于检索量子态的密度矩阵，特别是对于受限测量数据的情况表现良好。

    

    由于其良好的表达能力，神经网络一直被用于量子态重构（QST）。为了进一步提高重构量子态的效率，本文探讨了语言建模与量子态重构之间的相似性，并提出了一种基于注意力机制和变压器网络的 QST 方法，用于捕捉不同测量之间的相关性。我们的方法直接从测量统计数据中检索量子态的密度矩阵，并辅助使用综合损失函数来帮助最小化实际态与检索态之间的差异。然后，我们系统地跟踪了涉及各种参数调整的常见训练策略对基于注意力机制的 QST 方法的不同影响。结合这些技术，我们建立了一个稳健的基准线，可以有效地重构纯态和混合态。此外，通过比较三种不同的神经网络方法的性能，我们证明了我们的基于注意力机制的方法表现优于其他方法，特别是对于受限测量数据的情况。

    Neural networks have been actively explored for quantum state tomography (QST) due to their favorable expressibility. To further enhance the efficiency of reconstructing quantum states, we explore the similarity between language modeling and quantum state tomography and propose an attention-based QST method that utilizes the Transformer network to capture the correlations between measured results from different measurements. Our method directly retrieves the density matrices of quantum states from measured statistics, with the assistance of an integrated loss function that helps minimize the difference between the actual states and the retrieved states. Then, we systematically trace different impacts within a bag of common training strategies involving various parameter adjustments on the attention-based QST method. Combining these techniques, we establish a robust baseline that can efficiently reconstruct pure and mixed quantum states. Furthermore, by comparing the performance of thre
    

