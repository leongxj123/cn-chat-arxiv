# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Variational Inference of Parameters in Opinion Dynamics Models](https://arxiv.org/abs/2403.05358) | 通过将估计问题转化为可直接解决的优化任务，本研究提出了一种使用变分推断来估计意见动态ABM参数的方法。 |

# 详细

[^1]: 意见动态模型中参数的变分推断

    Variational Inference of Parameters in Opinion Dynamics Models

    [https://arxiv.org/abs/2403.05358](https://arxiv.org/abs/2403.05358)

    通过将估计问题转化为可直接解决的优化任务，本研究提出了一种使用变分推断来估计意见动态ABM参数的方法。

    

    尽管基于代理人的模型（ABMs）在研究社会现象中被频繁使用，但参数估计仍然是一个挑战，通常依赖于昂贵的基于模拟的启发式方法。本研究利用变分推断来估计意见动态ABM的参数，通过将估计问题转化为可直接解决的优化任务。

    arXiv:2403.05358v1 Announce Type: cross  Abstract: Despite the frequent use of agent-based models (ABMs) for studying social phenomena, parameter estimation remains a challenge, often relying on costly simulation-based heuristics. This work uses variational inference to estimate the parameters of an opinion dynamics ABM, by transforming the estimation problem into an optimization task that can be solved directly.   Our proposal relies on probabilistic generative ABMs (PGABMs): we start by synthesizing a probabilistic generative model from the ABM rules. Then, we transform the inference process into an optimization problem suitable for automatic differentiation. In particular, we use the Gumbel-Softmax reparameterization for categorical agent attributes and stochastic variational inference for parameter estimation. Furthermore, we explore the trade-offs of using variational distributions with different complexity: normal distributions and normalizing flows.   We validate our method on a
    

