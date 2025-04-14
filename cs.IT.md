# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Block-Level Draft Verification for Accelerating Speculative Decoding](https://arxiv.org/abs/2403.10444) | 提出了一种更好的草稿验证算法，通过将验证步骤制定为块级最优传输问题，实现了额外的墙钟速度提升，而不增加额外的计算成本和草稿标记 |

# 详细

[^1]: 用于加速推测解码的最佳块级草稿验证

    Optimal Block-Level Draft Verification for Accelerating Speculative Decoding

    [https://arxiv.org/abs/2403.10444](https://arxiv.org/abs/2403.10444)

    提出了一种更好的草稿验证算法，通过将验证步骤制定为块级最优传输问题，实现了额外的墙钟速度提升，而不增加额外的计算成本和草稿标记

    

    推测解码已被证明是在推理过程中加速大型语言模型（LLMs）无损加速的有效方法。 在每次迭代中，算法首先使用一个较小的模型起草一块标记。这些标记然后由大型模型并行验证，只有一部分标记将被保留，以确保最终输出遵循大型模型的分布。 在以往的所有推测解码工作中，起草验证是独立地逐个标记执行的。 在本工作中，我们提出了一个更好的起草验证算法，可提供额外的墙钟加速，而不需要额外的计算成本和起草标记。 我们首先将起草验证步骤制定为一个块级最优传输问题。 块级制定允许我们考虑更广泛的起草验证算法，并在一个起草中预期获得更多接受的标记数量

    arXiv:2403.10444v1 Announce Type: cross  Abstract: Speculative decoding has shown to be an effective method for lossless acceleration of large language models (LLMs) during inference. In each iteration, the algorithm first uses a smaller model to draft a block of tokens. The tokens are then verified by the large model in parallel and only a subset of tokens will be kept to guarantee that the final output follows the distribution of the large model. In all of the prior speculative decoding works, the draft verification is performed token-by-token independently. In this work, we propose a better draft verification algorithm that provides additional wall-clock speedup without incurring additional computation cost and draft tokens. We first formulate the draft verification step as a block-level optimal transport problem. The block-level formulation allows us to consider a wider range of draft verification algorithms and obtain a higher number of accepted tokens in expectation in one draft 
    

