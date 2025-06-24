# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Bayesian Non-parametric Approach to Generative Models: Integrating Variational Autoencoder and Generative Adversarial Networks using Wasserstein and Maximum Mean Discrepancy.](http://arxiv.org/abs/2308.14048) | 本研究提出了一种融合生成对抗网络和变分自编码器的贝叶斯非参数方法，通过在损失函数中使用Wasserstein和最大均值差异度量，实现了对潜在空间的有效学习，并能够生成多样且高质量的样本。 |
| [^2] | [Structural restrictions in local causal discovery: identifying direct causes of a target variable.](http://arxiv.org/abs/2307.16048) | 这项研究的目标是从观测数据中识别目标变量的直接原因，通过不对其他变量做太多假设，研究者提出了可识别性结果和两种实用算法。 |

# 详细

[^1]: 一种贝叶斯非参数方法用于生成模型：使用Wasserstein和最大均值差异度量集成变分自编码器和生成对抗网络

    A Bayesian Non-parametric Approach to Generative Models: Integrating Variational Autoencoder and Generative Adversarial Networks using Wasserstein and Maximum Mean Discrepancy. (arXiv:2308.14048v1 [stat.ML])

    [http://arxiv.org/abs/2308.14048](http://arxiv.org/abs/2308.14048)

    本研究提出了一种融合生成对抗网络和变分自编码器的贝叶斯非参数方法，通过在损失函数中使用Wasserstein和最大均值差异度量，实现了对潜在空间的有效学习，并能够生成多样且高质量的样本。

    

    生成模型已成为一种产生与真实图像难以区分的高质量图像的有前途的技术。生成对抗网络（GAN）和变分自编码器（VAE）是最为重要且被广泛研究的两种生成模型。GAN在生成逼真图像方面表现出色，而VAE则能够生成多样的图像。然而，GAN忽视了大部分可能的输出空间，这导致不能完全体现目标分布的多样性，而VAE则常常生成模糊图像。为了充分发挥两种模型的优点并减轻它们的弱点，我们采用了贝叶斯非参数方法将GAN和VAE相结合。我们的方法在损失函数中同时使用了Wasserstein和最大均值差异度量，以有效学习潜在空间并生成多样且高质量的样本。

    Generative models have emerged as a promising technique for producing high-quality images that are indistinguishable from real images. Generative adversarial networks (GANs) and variational autoencoders (VAEs) are two of the most prominent and widely studied generative models. GANs have demonstrated excellent performance in generating sharp realistic images and VAEs have shown strong abilities to generate diverse images. However, GANs suffer from ignoring a large portion of the possible output space which does not represent the full diversity of the target distribution, and VAEs tend to produce blurry images. To fully capitalize on the strengths of both models while mitigating their weaknesses, we employ a Bayesian non-parametric (BNP) approach to merge GANs and VAEs. Our procedure incorporates both Wasserstein and maximum mean discrepancy (MMD) measures in the loss function to enable effective learning of the latent space and generate diverse and high-quality samples. By fusing the di
    
[^2]: 局部因果发现中的结构限制: 识别目标变量的直接原因

    Structural restrictions in local causal discovery: identifying direct causes of a target variable. (arXiv:2307.16048v1 [stat.ME])

    [http://arxiv.org/abs/2307.16048](http://arxiv.org/abs/2307.16048)

    这项研究的目标是从观测数据中识别目标变量的直接原因，通过不对其他变量做太多假设，研究者提出了可识别性结果和两种实用算法。

    

    我们考虑从观察联合分布中学习目标变量的一组直接原因的问题。学习表示因果结构的有向无环图(DAG)是科学中的一个基本问题。当完整的DAG从分布中可识别时，已知有一些结果，例如假设非线性高斯数据生成过程。通常，我们只对识别一个目标变量的直接原因（局部因果结构），而不是完整的DAG感兴趣。在本文中，我们讨论了对目标变量的数据生成过程的不同假设，该假设下直接原因集合可以从分布中识别出来。在这样做的过程中，我们对除目标变量之外的变量基本上没有任何假设。除了新的可识别性结果，我们还提供了两种从有限随机样本估计直接原因的实用算法，并在几个基准数据集上证明了它们的有效性。

    We consider the problem of learning a set of direct causes of a target variable from an observational joint distribution. Learning directed acyclic graphs (DAGs) that represent the causal structure is a fundamental problem in science. Several results are known when the full DAG is identifiable from the distribution, such as assuming a nonlinear Gaussian data-generating process. Often, we are only interested in identifying the direct causes of one target variable (local causal structure), not the full DAG. In this paper, we discuss different assumptions for the data-generating process of the target variable under which the set of direct causes is identifiable from the distribution. While doing so, we put essentially no assumptions on the variables other than the target variable. In addition to the novel identifiability results, we provide two practical algorithms for estimating the direct causes from a finite random sample and demonstrate their effectiveness on several benchmark dataset
    

