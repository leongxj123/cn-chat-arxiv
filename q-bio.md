# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Disentangling Hippocampal Shape Variations: A Study of Neurological Disorders Using Graph Variational Autoencoder with Contrastive Learning](https://arxiv.org/abs/2404.00785) | 本研究利用图变分自动编码器和对比学习解开神经系统疾病中海马形状变异的关键潜变量，超越了其他先进方法在解开能力上的表现。 |
| [^2] | [Enabling Efficient Equivariant Operations in the Fourier Basis via Gaunt Tensor Products.](http://arxiv.org/abs/2401.10216) | 该论文提出了一种加速计算不可约表示张量积的方法，通过将等变操作基础从球形谐波改变为2D傅立叶基础，实现了对E(3)群的等变神经网络的高效建模。 |

# 详细

[^1]: 解开海马形状变异之谜：利用对比学习的图变分自动编码器研究神经系统疾病

    Disentangling Hippocampal Shape Variations: A Study of Neurological Disorders Using Graph Variational Autoencoder with Contrastive Learning

    [https://arxiv.org/abs/2404.00785](https://arxiv.org/abs/2404.00785)

    本研究利用图变分自动编码器和对比学习解开神经系统疾病中海马形状变异的关键潜变量，超越了其他先进方法在解开能力上的表现。

    

    本文提出了一项综合研究，专注于在神经系统疾病背景下从扩散张量成像（DTI）数据集中解开海马形状变异。借助增强的监督对比学习图变分自动编码器（VAE），我们的方法旨在通过区分代表年龄和是否患病的两个不同潜变量来提高解释性。在我们的消融研究中，我们调查了一系列VAE架构和对比损失函数，展示了我们方法增强的解开能力。这个评估使用了来自DTI海马数据集的合成3D环形网格数据和真实的3D海马网格数据集。我们的监督解开模型在解开分数方面优于几种最先进的方法，如属性和引导VAE。我们的模型可以区分不同年龄组和疾病状况。

    arXiv:2404.00785v1 Announce Type: cross  Abstract: This paper presents a comprehensive study focused on disentangling hippocampal shape variations from diffusion tensor imaging (DTI) datasets within the context of neurological disorders. Leveraging a Graph Variational Autoencoder (VAE) enhanced with Supervised Contrastive Learning, our approach aims to improve interpretability by disentangling two distinct latent variables corresponding to age and the presence of diseases. In our ablation study, we investigate a range of VAE architectures and contrastive loss functions, showcasing the enhanced disentanglement capabilities of our approach. This evaluation uses synthetic 3D torus mesh data and real 3D hippocampal mesh datasets derived from the DTI hippocampal dataset. Our supervised disentanglement model outperforms several state-of-the-art (SOTA) methods like attribute and guided VAEs in terms of disentanglement scores. Our model distinguishes between age groups and disease status in pa
    
[^2]: 通过Gaunt张量积在傅里叶基础上实现高效的等变操作

    Enabling Efficient Equivariant Operations in the Fourier Basis via Gaunt Tensor Products. (arXiv:2401.10216v1 [cs.LG])

    [http://arxiv.org/abs/2401.10216](http://arxiv.org/abs/2401.10216)

    该论文提出了一种加速计算不可约表示张量积的方法，通过将等变操作基础从球形谐波改变为2D傅立叶基础，实现了对E(3)群的等变神经网络的高效建模。

    

    在建模现实世界应用中的3D数据时，发展E(3)群的等变神经网络起着重要作用。实现这种等变性主要涉及到不可约表示（irreps）的张量积。然而，随着使用高阶张量，这些操作的计算复杂性显著增加。在这项工作中，我们提出了一种系统的方法来大大加速不可约表示的张量积的计算。我们将常用的Clebsch-Gordan系数与Gaunt系数进行了数学上的连接，Gaunt系数是三个球形谐波乘积的积分。通过Gaunt系数，不可约表示的张量积等价于由球形谐波表示的球形函数之间的乘法。这种观点进一步使我们能够将等变操作的基础从球形谐波改变为2D傅立叶基础。因此，球形函数之间的乘法可以在傅立叶基础上进行。

    Developing equivariant neural networks for the E(3) group plays an important role in modeling 3D data across real-world applications. Enforcing this equivariance primarily involves the tensor products of irreducible representations (irreps). However, the computational complexity of such operations increases significantly as higher-order tensors are used. In this work, we propose a systematic approach to substantially accelerate the computation of the tensor products of irreps. We mathematically connect the commonly used Clebsch-Gordan coefficients to the Gaunt coefficients, which are integrals of products of three spherical harmonics. Through Gaunt coefficients, the tensor product of irreps becomes equivalent to the multiplication between spherical functions represented by spherical harmonics. This perspective further allows us to change the basis for the equivariant operations from spherical harmonics to a 2D Fourier basis. Consequently, the multiplication between spherical functions 
    

