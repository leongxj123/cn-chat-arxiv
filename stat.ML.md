# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AdAdaGrad: Adaptive Batch Size Schemes for Adaptive Gradient Methods](https://arxiv.org/abs/2402.11215) | AdAdaGrad和AdAdaGradNorm是一个自适应增加批大小的方法，在深度学习中引入了自适应批大小策略，证明AdaGradNorm以高概率在$O(1/K)$速度下收敛。 |
| [^2] | [A Discriminative Latent-Variable Model for Bilingual Lexicon Induction](https://arxiv.org/abs/1808.09334) | 引入判别式潜变量模型，结合先前研究的词典先验和表示法，提出了用于双语词典归纳的新方法，并通过实验证据展示先验可以改善诱导的双语词典。 |

# 详细

[^1]: AdAdaGrad：自适应梯度方法的自适应批大小方案

    AdAdaGrad: Adaptive Batch Size Schemes for Adaptive Gradient Methods

    [https://arxiv.org/abs/2402.11215](https://arxiv.org/abs/2402.11215)

    AdAdaGrad和AdAdaGradNorm是一个自适应增加批大小的方法，在深度学习中引入了自适应批大小策略，证明AdaGradNorm以高概率在$O(1/K)$速度下收敛。

    

    随机梯度优化器中批量大小的选择对模型训练至关重要。然而，在训练过程中变化批大小的实践相对其他超参数较少探讨。我们研究了从自适应采样方法中导出的自适应批大小策略，传统上仅应用于随机梯度下降。考虑到学习速率和批大小之间的显著相互作用，以及自适应梯度方法在深度学习中的普及，我们强调在这些情境中需要自适应批大小策略。我们介绍了AdAdaGrad及其标量变体AdAdaGradNorm，它们在训练过程中逐渐增加批大小，同时使用AdaGrad和AdaGradNorm进行模型更新。我们证明了AdaGradNorm以高概率以$O(1/K)$的速度收敛，用于找到光滑非凸函数的一阶稳定点在$K$次迭代内。

    arXiv:2402.11215v1 Announce Type: new  Abstract: The choice of batch sizes in stochastic gradient optimizers is critical for model training. However, the practice of varying batch sizes throughout the training process is less explored compared to other hyperparameters. We investigate adaptive batch size strategies derived from adaptive sampling methods, traditionally applied only in stochastic gradient descent. Given the significant interplay between learning rates and batch sizes, and considering the prevalence of adaptive gradient methods in deep learning, we emphasize the need for adaptive batch size strategies in these contexts. We introduce AdAdaGrad and its scalar variant AdAdaGradNorm, which incrementally increase batch sizes during training, while model updates are performed using AdaGrad and AdaGradNorm. We prove that AdaGradNorm converges with high probability at a rate of $\mathscr{O}(1/K)$ for finding a first-order stationary point of smooth nonconvex functions within $K$ i
    
[^2]: 一种用于双语词典归纳的判别式潜变量模型

    A Discriminative Latent-Variable Model for Bilingual Lexicon Induction

    [https://arxiv.org/abs/1808.09334](https://arxiv.org/abs/1808.09334)

    引入判别式潜变量模型，结合先前研究的词典先验和表示法，提出了用于双语词典归纳的新方法，并通过实验证据展示先验可以改善诱导的双语词典。

    

    我们引入了一种新颖的用于双语词典归纳的判别式潜变量模型。我们的模型将Haghighi等人（2008）的二分匹配词典先验与基于表示的方法（Artetxe等人，2017）相结合。为了训练模型，我们推导出了高效的Viterbi EM算法。我们在两个度量标准下对六种语言对进行了实证结果，并显示先验改善了诱导的双语词典。我们还演示了如何将先前的工作视为类似风格的潜变量模型，尽管有不同的先验。

    arXiv:1808.09334v3 Announce Type: replace  Abstract: We introduce a novel discriminative latent variable model for bilingual lexicon induction. Our model combines the bipartite matching dictionary prior of Haghighi et al. (2008) with a representation-based approach (Artetxe et al., 2017). To train the model, we derive an efficient Viterbi EM algorithm. We provide empirical results on six language pairs under two metrics and show that the prior improves the induced bilingual lexicons. We also demonstrate how previous work may be viewed as a similarly fashioned latent-variable model, albeit with a different prior.
    

