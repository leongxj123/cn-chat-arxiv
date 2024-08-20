# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Augmented Online Facility Location.](http://arxiv.org/abs/2107.08277) | 本文提出了一种用于在线设施选址问题的在线算法，竞争比率能够光滑地随误差减小而从对数级别降低到常数级别。 |

# 详细

[^1]: 学习增强的在线设施选址问题

    Learning Augmented Online Facility Location. (arXiv:2107.08277v3 [cs.DS] UPDATED)

    [http://arxiv.org/abs/2107.08277](http://arxiv.org/abs/2107.08277)

    本文提出了一种用于在线设施选址问题的在线算法，竞争比率能够光滑地随误差减小而从对数级别降低到常数级别。

    

    本文考虑了在线设施选址问题，在该问题中，需求逐一到达，并且必须在到达时将其（无法撤销地）分配给一个开放的设施，而没有关于未来需求的任何知识。我们提出了一种在线算法，用于处理该问题，并利用对最优设施位置的预测。我们证明了，竞争比率从需求数量的对数级别平滑地降低到常数级别，当错误（即预测位置与最优设施位置之间的总距离）向零降低时。我们配合算法的降低界，建立了算法的竞争比率对误差的依赖关系是最优的，常数接近最优。

    Following the research agenda initiated by Munoz & Vassilvitskii [1] and Lykouris & Vassilvitskii [2] on learning-augmented online algorithms for classical online optimization problems, in this work, we consider the Online Facility Location problem under this framework. In Online Facility Location (OFL), demands arrive one-by-one in a metric space and must be (irrevocably) assigned to an open facility upon arrival, without any knowledge about future demands.  We present an online algorithm for OFL that exploits potentially imperfect predictions on the locations of the optimal facilities. We prove that the competitive ratio decreases smoothly from sublogarithmic in the number of demands to constant, as the error, i.e., the total distance of the predicted locations to the optimal facility locations, decreases towards zero. We complement our analysis with a matching lower bound establishing that the dependence of the algorithm's competitive ratio on the error is optimal, up to constant fa
    

