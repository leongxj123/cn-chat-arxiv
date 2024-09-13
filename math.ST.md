# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Score diffusion models without early stopping: finite Fisher information is all you need.](http://arxiv.org/abs/2308.12240) | 无早停的分数扩散模型不需要得分函数的Lipschitz均匀条件，只需要有限的费舍尔信息。 |

# 详细

[^1]: 无早停的分数扩散模型：有限费舍尔信息就足够了

    Score diffusion models without early stopping: finite Fisher information is all you need. (arXiv:2308.12240v1 [math.ST])

    [http://arxiv.org/abs/2308.12240](http://arxiv.org/abs/2308.12240)

    无早停的分数扩散模型不需要得分函数的Lipschitz均匀条件，只需要有限的费舍尔信息。

    

    分数扩散模型是一种围绕着与随机微分方程相关的得分函数估计的生成模型。在获得近似的得分函数之后，利用它来模拟相应的时间逆过程，最终实现近似数据样本的生成。尽管这些模型具有显著的实际意义，但在涉及非常规得分和估计器的情况下，仍存在一个显著的挑战，即缺乏全面的定量结果。在几乎所有的Kullback Leibler散度的相关结果中，都假设得分函数或其近似在时间上是Lipschitz均匀的。然而，在实践中，这个条件非常严格，或者很难建立。为了解决这个问题，先前的研究主要是关注分数扩散模型的早停版本在KL散度上的收敛界限，并且...

    Diffusion models are a new class of generative models that revolve around the estimation of the score function associated with a stochastic differential equation. Subsequent to its acquisition, the approximated score function is then harnessed to simulate the corresponding time-reversal process, ultimately enabling the generation of approximate data samples. Despite their evident practical significance these models carry, a notable challenge persists in the form of a lack of comprehensive quantitative results, especially in scenarios involving non-regular scores and estimators. In almost all reported bounds in Kullback Leibler (KL) divergence, it is assumed that either the score function or its approximation is Lipschitz uniformly in time. However, this condition is very restrictive in practice or appears to be difficult to establish.  To circumvent this issue, previous works mainly focused on establishing convergence bounds in KL for an early stopped version of the diffusion model and
    

