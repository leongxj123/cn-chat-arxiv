# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Interventional and Counterfactual Inference with Diffusion Models.](http://arxiv.org/abs/2302.00860) | 本论文提出了基于扩散模型的因果模型 (DCM)，它可以在只有观测数据和因果图可用的情况下进行干预和反事实推断，其具有较好的表现。同时，论文还提供了一种分析反事实估计的方法，可以应用于更广泛的场景。 |

# 详细

[^1]: 利用扩散模型进行干预和反事实推断

    Interventional and Counterfactual Inference with Diffusion Models. (arXiv:2302.00860v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.00860](http://arxiv.org/abs/2302.00860)

    本论文提出了基于扩散模型的因果模型 (DCM)，它可以在只有观测数据和因果图可用的情况下进行干预和反事实推断，其具有较好的表现。同时，论文还提供了一种分析反事实估计的方法，可以应用于更广泛的场景。

    

    我们考虑在只有观测数据和因果图可用的因果充分设置中回答观测、干预和反事实查询的问题。利用扩散模型的最新发展，我们引入了基于扩散的因果模型 (DCM)，来学习生成独特的潜在编码的因果机制。这些编码使我们能够在干预下直接采样和进行反事实推断。扩散模型在这里是一个自然的选择，因为它们可以将每个节点编码为一个代表外生噪声的潜在表示。我们的实证评估表明，在回答因果查询方面，与现有的最先进方法相比，有显着的改进。此外，我们提供了理论结果，为分析一般编码器-解码器模型中的反事实估计提供一种方法，这对我们提出的方法以外的设置可能也有用。

    We consider the problem of answering observational, interventional, and counterfactual queries in a causally sufficient setting where only observational data and the causal graph are available. Utilizing the recent developments in diffusion models, we introduce diffusion-based causal models (DCM) to learn causal mechanisms, that generate unique latent encodings. These encodings enable us to directly sample under interventions and perform abduction for counterfactuals. Diffusion models are a natural fit here, since they can encode each node to a latent representation that acts as a proxy for exogenous noise. Our empirical evaluations demonstrate significant improvements over existing state-of-the-art methods for answering causal queries. Furthermore, we provide theoretical results that offer a methodology for analyzing counterfactual estimation in general encoder-decoder models, which could be useful in settings beyond our proposed approach.
    

