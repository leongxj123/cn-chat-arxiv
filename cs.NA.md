# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Integral Operator Approaches for Scattered Data Fitting on Spheres.](http://arxiv.org/abs/2401.15294) | 本文提出了一种积分算子方法来解决球面上的散点数据拟合问题，通过研究加权谱滤波算法的逼近性能，成功推导出了带权重谱滤波算法的最优误差估计。这种方法可以避免一些现有方法中存在的问题，同时提供了一种优化算法的解决方案。 |
| [^2] | [Stable generative modeling using diffusion maps.](http://arxiv.org/abs/2401.04372) | 本文提出了一种稳定的生成建模方法，通过将扩散映射与朗之万动力学相结合，在仅有有限数量的训练样本的情况下生成新样本，并解决了时间步长僵硬随机微分方程中的稳定性问题。 |

# 详细

[^1]: 球面上散点数据拟合的积分算子方法

    Integral Operator Approaches for Scattered Data Fitting on Spheres. (arXiv:2401.15294v1 [math.NA])

    [http://arxiv.org/abs/2401.15294](http://arxiv.org/abs/2401.15294)

    本文提出了一种积分算子方法来解决球面上的散点数据拟合问题，通过研究加权谱滤波算法的逼近性能，成功推导出了带权重谱滤波算法的最优误差估计。这种方法可以避免一些现有方法中存在的问题，同时提供了一种优化算法的解决方案。

    

    本文着重研究了球面上的散点数据拟合问题。我们研究了一类加权谱滤波算法（包括Tikhonov正则化、Landaweber迭代、谱截断和迭代Tikhonov）在拟合可能存在的无界随机噪声的嘈杂数据时的逼近性能。为了分析这个问题，我们提出了一种积分算子方法，可以被看作是散点数据拟合领域中广泛使用的采样不等式方法和规范集方法的延伸。通过提供算子差异和数值积分规则之间的等价性，我们成功地推导出带权重谱滤波算法的Sobolev类型误差估计的最优结果。我们的误差估计不受文献中Tikhonov正则化的饱和现象、现有误差分析中的本地空间屏障和不同嵌入空间的影响。我们还提出了一种分而治之的方案，以提升加权谱滤波算法的效能。

    This paper focuses on scattered data fitting problems on spheres. We study the approximation performance of a class of weighted spectral filter algorithms, including Tikhonov regularization, Landaweber iteration, spectral cut-off, and iterated Tikhonov, in fitting noisy data with possibly unbounded random noise. For the analysis, we develop an integral operator approach that can be regarded as an extension of the widely used sampling inequality approach and norming set method in the community of scattered data fitting. After providing an equivalence between the operator differences and quadrature rules, we succeed in deriving optimal Sobolev-type error estimates of weighted spectral filter algorithms. Our derived error estimates do not suffer from the saturation phenomenon for Tikhonov regularization in the literature, native-space-barrier for existing error analysis and adapts to different embedding spaces. We also propose a divide-and-conquer scheme to equip weighted spectral filter 
    
[^2]: 使用扩散映射进行稳定的生成建模

    Stable generative modeling using diffusion maps. (arXiv:2401.04372v1 [stat.ML])

    [http://arxiv.org/abs/2401.04372](http://arxiv.org/abs/2401.04372)

    本文提出了一种稳定的生成建模方法，通过将扩散映射与朗之万动力学相结合，在仅有有限数量的训练样本的情况下生成新样本，并解决了时间步长僵硬随机微分方程中的稳定性问题。

    

    我们考虑从仅有足够数量的训练样本可得到的未知分布中抽样的问题。在生成建模的背景下，这样的设置最近引起了相当大的关注。本文中，我们提出了一种将扩散映射和朗之万动力学相结合的生成模型。扩散映射用于从可用的训练样本中近似得到漂移项，然后在离散时间的朗之万采样器中实现生成新样本。通过将核带宽设置为与未调整的朗之万算法中使用的时间步长匹配，我们的方法可以有效地避免通常与时间步长僵硬随机微分方程相关的稳定性问题。更准确地说，我们引入了一种新颖的分裂步骤方案，确保生成的样本保持在训练样本的凸包内。我们的框架可以自然地扩展为生成条件样本。我们展示了性能。

    We consider the problem of sampling from an unknown distribution for which only a sufficiently large number of training samples are available. Such settings have recently drawn considerable interest in the context of generative modelling. In this paper, we propose a generative model combining diffusion maps and Langevin dynamics. Diffusion maps are used to approximate the drift term from the available training samples, which is then implemented in a discrete-time Langevin sampler to generate new samples. By setting the kernel bandwidth to match the time step size used in the unadjusted Langevin algorithm, our method effectively circumvents any stability issues typically associated with time-stepping stiff stochastic differential equations. More precisely, we introduce a novel split-step scheme, ensuring that the generated samples remain within the convex hull of the training samples. Our framework can be naturally extended to generate conditional samples. We demonstrate the performance
    

