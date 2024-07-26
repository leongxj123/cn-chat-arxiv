# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Block-Level Draft Verification for Accelerating Speculative Decoding](https://arxiv.org/abs/2403.10444) | 提出了一种更好的草稿验证算法，通过将验证步骤制定为块级最优传输问题，实现了额外的墙钟速度提升，而不增加额外的计算成本和草稿标记 |
| [^2] | [Detection of Correlated Random Vectors.](http://arxiv.org/abs/2401.13429) | 本文研究了判断两个标准正态随机向量是否相关的问题，提出了一种新的方法来评估似然比的二阶矩，并发现了与整数分割函数之间的联系。 |

# 详细

[^1]: 用于加速推测解码的最佳块级草稿验证

    Optimal Block-Level Draft Verification for Accelerating Speculative Decoding

    [https://arxiv.org/abs/2403.10444](https://arxiv.org/abs/2403.10444)

    提出了一种更好的草稿验证算法，通过将验证步骤制定为块级最优传输问题，实现了额外的墙钟速度提升，而不增加额外的计算成本和草稿标记

    

    推测解码已被证明是在推理过程中加速大型语言模型（LLMs）无损加速的有效方法。 在每次迭代中，算法首先使用一个较小的模型起草一块标记。这些标记然后由大型模型并行验证，只有一部分标记将被保留，以确保最终输出遵循大型模型的分布。 在以往的所有推测解码工作中，起草验证是独立地逐个标记执行的。 在本工作中，我们提出了一个更好的起草验证算法，可提供额外的墙钟加速，而不需要额外的计算成本和起草标记。 我们首先将起草验证步骤制定为一个块级最优传输问题。 块级制定允许我们考虑更广泛的起草验证算法，并在一个起草中预期获得更多接受的标记数量

    arXiv:2403.10444v1 Announce Type: cross  Abstract: Speculative decoding has shown to be an effective method for lossless acceleration of large language models (LLMs) during inference. In each iteration, the algorithm first uses a smaller model to draft a block of tokens. The tokens are then verified by the large model in parallel and only a subset of tokens will be kept to guarantee that the final output follows the distribution of the large model. In all of the prior speculative decoding works, the draft verification is performed token-by-token independently. In this work, we propose a better draft verification algorithm that provides additional wall-clock speedup without incurring additional computation cost and draft tokens. We first formulate the draft verification step as a block-level optimal transport problem. The block-level formulation allows us to consider a wider range of draft verification algorithms and obtain a higher number of accepted tokens in expectation in one draft 
    
[^2]: 相关随机向量的检测

    Detection of Correlated Random Vectors. (arXiv:2401.13429v1 [cs.IT])

    [http://arxiv.org/abs/2401.13429](http://arxiv.org/abs/2401.13429)

    本文研究了判断两个标准正态随机向量是否相关的问题，提出了一种新的方法来评估似然比的二阶矩，并发现了与整数分割函数之间的联系。

    

    在本文中，我们研究了判断两个标准正态随机向量$\mathsf{X}\in\mathbb{R}^{n}$和$\mathsf{Y}\in\mathbb{R}^{n}$是否相关的问题。这被表述为一个假设检验问题，在零假设下，这些向量是统计独立的，而在备择假设下，$\mathsf{X}$和随机均匀置换的$\mathsf{Y}$是具有相关系数$\rho$的。我们分析了信息论上不可能和可能的最优测试阈值，作为$n$和$\rho$的函数。为了得出我们的信息论下界，我们开发了一种利用正交多项式展开来评估似然比的二阶矩的新技术，该技术揭示了与整数分割函数之间的一个令人惊讶的联系。我们还研究了上述设置的多维泛化，其中我们观察到两个数据库/矩阵，而不是两个向量。

    In this paper, we investigate the problem of deciding whether two standard normal random vectors $\mathsf{X}\in\mathbb{R}^{n}$ and $\mathsf{Y}\in\mathbb{R}^{n}$ are correlated or not. This is formulated as a hypothesis testing problem, where under the null hypothesis, these vectors are statistically independent, while under the alternative, $\mathsf{X}$ and a randomly and uniformly permuted version of $\mathsf{Y}$, are correlated with correlation $\rho$. We analyze the thresholds at which optimal testing is information-theoretically impossible and possible, as a function of $n$ and $\rho$. To derive our information-theoretic lower bounds, we develop a novel technique for evaluating the second moment of the likelihood ratio using an orthogonal polynomials expansion, which among other things, reveals a surprising connection to integer partition functions. We also study a multi-dimensional generalization of the above setting, where rather than two vectors we observe two databases/matrices
    

