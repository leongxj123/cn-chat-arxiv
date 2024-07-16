# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stable generative modeling using diffusion maps.](http://arxiv.org/abs/2401.04372) | 本文提出了一种稳定的生成建模方法，通过将扩散映射与朗之万动力学相结合，在仅有有限数量的训练样本的情况下生成新样本，并解决了时间步长僵硬随机微分方程中的稳定性问题。 |

# 详细

[^1]: 使用扩散映射进行稳定的生成建模

    Stable generative modeling using diffusion maps. (arXiv:2401.04372v1 [stat.ML])

    [http://arxiv.org/abs/2401.04372](http://arxiv.org/abs/2401.04372)

    本文提出了一种稳定的生成建模方法，通过将扩散映射与朗之万动力学相结合，在仅有有限数量的训练样本的情况下生成新样本，并解决了时间步长僵硬随机微分方程中的稳定性问题。

    

    我们考虑从仅有足够数量的训练样本可得到的未知分布中抽样的问题。在生成建模的背景下，这样的设置最近引起了相当大的关注。本文中，我们提出了一种将扩散映射和朗之万动力学相结合的生成模型。扩散映射用于从可用的训练样本中近似得到漂移项，然后在离散时间的朗之万采样器中实现生成新样本。通过将核带宽设置为与未调整的朗之万算法中使用的时间步长匹配，我们的方法可以有效地避免通常与时间步长僵硬随机微分方程相关的稳定性问题。更准确地说，我们引入了一种新颖的分裂步骤方案，确保生成的样本保持在训练样本的凸包内。我们的框架可以自然地扩展为生成条件样本。我们展示了性能。

    We consider the problem of sampling from an unknown distribution for which only a sufficiently large number of training samples are available. Such settings have recently drawn considerable interest in the context of generative modelling. In this paper, we propose a generative model combining diffusion maps and Langevin dynamics. Diffusion maps are used to approximate the drift term from the available training samples, which is then implemented in a discrete-time Langevin sampler to generate new samples. By setting the kernel bandwidth to match the time step size used in the unadjusted Langevin algorithm, our method effectively circumvents any stability issues typically associated with time-stepping stiff stochastic differential equations. More precisely, we introduce a novel split-step scheme, ensuring that the generated samples remain within the convex hull of the training samples. Our framework can be naturally extended to generate conditional samples. We demonstrate the performance
    

