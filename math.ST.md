# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Statistical Properties of Generative Adversarial Models for Low Intrinsic Data Dimension.](http://arxiv.org/abs/2401.15801) | 这篇论文研究了用于低固有数据维度的生成对抗模型的统计属性，提出了关于估计密度的统计保证，涉及数据和潜空间的内在维度，并证明了估计结果与目标的期望Wasserstein-1距离的缩放关系。 |

# 详细

[^1]: 关于用于低固有数据维度的生成对抗模型的统计属性

    On the Statistical Properties of Generative Adversarial Models for Low Intrinsic Data Dimension. (arXiv:2401.15801v1 [stat.ML])

    [http://arxiv.org/abs/2401.15801](http://arxiv.org/abs/2401.15801)

    这篇论文研究了用于低固有数据维度的生成对抗模型的统计属性，提出了关于估计密度的统计保证，涉及数据和潜空间的内在维度，并证明了估计结果与目标的期望Wasserstein-1距离的缩放关系。

    

    尽管生成对抗网络（GANs）取得了显著的实证成功，但其统计准确性的理论保证仍然相对悲观。特别是在应用GANs的数据分布（如自然图像）中，通常假设其在高维特征空间中具有固有的低维结构，但这在现有分析中往往没有得到反映。在本文中，我们试图通过推导关于数据和潜空间的内在维度的统计保证来弥合GANs及其双向变体BiGANs在理论和实践之间的差距。我们分析地证明，如果我们有来自未知目标分布的 n 个样本，并且选择了适当的网络架构，那么从目标中估计得出的期望 Wasserstein-1 距离会按照 $O(n^{-1/d_\mu })$ 缩放。

    Despite the remarkable empirical successes of Generative Adversarial Networks (GANs), the theoretical guarantees for their statistical accuracy remain rather pessimistic. In particular, the data distributions on which GANs are applied, such as natural images, are often hypothesized to have an intrinsic low-dimensional structure in a typically high-dimensional feature space, but this is often not reflected in the derived rates in the state-of-the-art analyses. In this paper, we attempt to bridge the gap between the theory and practice of GANs and their bidirectional variant, Bi-directional GANs (BiGANs), by deriving statistical guarantees on the estimated densities in terms of the intrinsic dimension of the data and the latent space. We analytically show that if one has access to $n$ samples from the unknown target distribution and the network architectures are properly chosen, the expected Wasserstein-1 distance of the estimates from the target scales as $O\left( n^{-1/d_\mu } \right)$
    

