# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Size-Independent Generalization Bounds for Deep Operator Nets.](http://arxiv.org/abs/2205.11359) | 本论文研究了深度操作器网络的泛化界限问题，在一类DeepONets中证明了它们的Rademacher复杂度的界限不会随网络宽度扩展而明确变化，并利用这个结果展示了如何选择Huber损失来获得不明确依赖于网络大小的泛化误差界限。 |

# 详细

[^1]: 面向尺度无关的深度操作器网络的泛化界限

    Towards Size-Independent Generalization Bounds for Deep Operator Nets. (arXiv:2205.11359v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2205.11359](http://arxiv.org/abs/2205.11359)

    本论文研究了深度操作器网络的泛化界限问题，在一类DeepONets中证明了它们的Rademacher复杂度的界限不会随网络宽度扩展而明确变化，并利用这个结果展示了如何选择Huber损失来获得不明确依赖于网络大小的泛化误差界限。

    

    在最近的时期，机器学习方法在分析物理系统方面取得了重要进展。在这个主题中特别活跃的领域是"物理信息机器学习"，它专注于使用神经网络来数值求解微分方程。在这项工作中，我们旨在推进在训练DeepONets时测量样本外误差的理论 - 这是解决PDE系统最通用的方法之一。首先，针对一类DeepONets，我们证明了它们的Rademacher复杂度有一个界限，该界限不会明确地随着涉及的网络宽度扩展。其次，我们利用这一结果来展示如何选择Huber损失，使得对于这些DeepONet类，能够获得不明确依赖于网络大小的泛化误差界限。我们指出，我们的理论结果适用于任何目标是由DeepONets求解的PDE。

    In recent times machine learning methods have made significant advances in becoming a useful tool for analyzing physical systems. A particularly active area in this theme has been "physics-informed machine learning" which focuses on using neural nets for numerically solving differential equations. In this work, we aim to advance the theory of measuring out-of-sample error while training DeepONets -- which is among the most versatile ways to solve PDE systems in one-shot.  Firstly, for a class of DeepONets, we prove a bound on their Rademacher complexity which does not explicitly scale with the width of the nets involved. Secondly, we use this to show how the Huber loss can be chosen so that for these DeepONet classes generalization error bounds can be obtained that have no explicit dependence on the size of the nets. We note that our theoretical results apply to any PDE being targeted to be solved by DeepONets.
    

