# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LogicMP: A Neuro-symbolic Approach for Encoding First-order Logic Constraints.](http://arxiv.org/abs/2309.15458) | 本文提出了一种名为LogicMP的新颖神经层，该层通过均场变分推断将一阶逻辑约束编码进神经网络中。通过有效缓解一阶逻辑模型的推断困难，LogicMP在图形、图像和文本任务中表现出比竞争对手更好的性能和效率。 |

# 详细

[^1]: LogicMP: 一种将一阶逻辑约束编码的神经符号方法

    LogicMP: A Neuro-symbolic Approach for Encoding First-order Logic Constraints. (arXiv:2309.15458v1 [cs.AI])

    [http://arxiv.org/abs/2309.15458](http://arxiv.org/abs/2309.15458)

    本文提出了一种名为LogicMP的新颖神经层，该层通过均场变分推断将一阶逻辑约束编码进神经网络中。通过有效缓解一阶逻辑模型的推断困难，LogicMP在图形、图像和文本任务中表现出比竞争对手更好的性能和效率。

    

    将一阶逻辑约束与神经网络集成是一个关键但具有挑战性的问题，因为它涉及建模复杂的相关性以满足约束。本文提出了一种新颖的神经层LogicMP，其层对MLN进行均场变分推断。它可以插入任何现成的神经网络以编码一阶逻辑约束，同时保持模块化和效率。通过利用MLN中的结构和对称性，我们从理论上证明了我们设计良好、高效的均场迭代能够有效缓解MLN推断的困难，将推断从顺序计算降低为一系列并行的张量操作。在图形、图像和文本的三类任务上的实证结果表明，LogicMP在性能和效率上都优于先进的竞争对手。

    Integrating first-order logic constraints (FOLCs) with neural networks is a crucial but challenging problem since it involves modeling intricate correlations to satisfy the constraints. This paper proposes a novel neural layer, LogicMP, whose layers perform mean-field variational inference over an MLN. It can be plugged into any off-the-shelf neural network to encode FOLCs while retaining modularity and efficiency. By exploiting the structure and symmetries in MLNs, we theoretically demonstrate that our well-designed, efficient mean-field iterations effectively mitigate the difficulty of MLN inference, reducing the inference from sequential calculation to a series of parallel tensor operations. Empirical results in three kinds of tasks over graphs, images, and text show that LogicMP outperforms advanced competitors in both performance and efficiency.
    

