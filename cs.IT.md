# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How Transformers Learn Causal Structure with Gradient Descent](https://arxiv.org/abs/2402.14735) | Transformers通过梯度下降学习因果结构的过程中，关键的证据是注意力矩阵的梯度编码了token之间的互信息 |

# 详细

[^1]: Transformers如何通过梯度下降学习因果结构

    How Transformers Learn Causal Structure with Gradient Descent

    [https://arxiv.org/abs/2402.14735](https://arxiv.org/abs/2402.14735)

    Transformers通过梯度下降学习因果结构的过程中，关键的证据是注意力矩阵的梯度编码了token之间的互信息

    

    Transformers在序列建模任务上取得了令人难以置信的成功，这在很大程度上归功于自注意机制，它允许信息在序列的不同部分之间传递。自注意机制使得transformers能够编码因果结构，从而使其特别适合序列建模。然而，transformers通过梯度训练算法学习这种因果结构的过程仍然不太清楚。为了更好地理解这个过程，我们引入了一个需要学习潜在因果结构的上下文学习任务。我们证明了简化的两层transformer上的梯度下降可以学会解决这个任务，通过在第一层注意力中编码潜在因果图来完成。我们证明的关键洞察是注意力矩阵的梯度编码了token之间的互信息。由于数据处理不等式的结果，注意力矩阵中最大的条目...

    arXiv:2402.14735v1 Announce Type: new  Abstract: The incredible success of transformers on sequence modeling tasks can be largely attributed to the self-attention mechanism, which allows information to be transferred between different parts of a sequence. Self-attention allows transformers to encode causal structure which makes them particularly suitable for sequence modeling. However, the process by which transformers learn such causal structure via gradient-based training algorithms remains poorly understood. To better understand this process, we introduce an in-context learning task that requires learning latent causal structure. We prove that gradient descent on a simplified two-layer transformer learns to solve this task by encoding the latent causal graph in the first attention layer. The key insight of our proof is that the gradient of the attention matrix encodes the mutual information between tokens. As a consequence of the data processing inequality, the largest entries of th
    

