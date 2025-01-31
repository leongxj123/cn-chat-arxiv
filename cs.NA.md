# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient Algorithms for Regularized Nonnegative Scale-invariant Low-rank Approximation Models](https://arxiv.org/abs/2403.18517) | 通过研究称为均匀正则化尺度不变的更一般模型，揭示了低秩逼近模型中尺度不变性导致隐式正则化的效果，有助于更好理解正则化函数的作用并指导正则化超参数的选择。 |
| [^2] | [Generative Adversarial Reduced Order Modelling.](http://arxiv.org/abs/2305.15881) | 本文提出了一种基于GAN的简化建模方法GAROM，通过引入一个数据驱动的生成对抗模型，能够学习参数微分方程的解，并获得了较好的实验效果。 |

# 详细

[^1]: 针对正则化非负尺度不变低秩逼近模型的高效算法

    Efficient Algorithms for Regularized Nonnegative Scale-invariant Low-rank Approximation Models

    [https://arxiv.org/abs/2403.18517](https://arxiv.org/abs/2403.18517)

    通过研究称为均匀正则化尺度不变的更一般模型，揭示了低秩逼近模型中尺度不变性导致隐式正则化的效果，有助于更好理解正则化函数的作用并指导正则化超参数的选择。

    

    正则化非负低秩逼近，如稀疏的非负矩阵分解或稀疏的非负Tucker分解，是具有增强可解释性的降维模型中的一个重要分支。然而，从实践角度来看，由于这些模型的多因素特性以及缺乏支持这些选择的理论，正则化函数和正则化系数的选择，以及高效算法的设计仍然具有挑战性。本文旨在改进这些问题。通过研究一个称为均匀正则化尺度不变的更一般模型，我们证明低秩逼近模型中固有的尺度不变性导致了隐式正则化，具有意想不到的有益和有害效果。这一发现使我们能够更好地理解低秩逼近模型中正则化函数的作用，指导正则化超参数的选择。

    arXiv:2403.18517v1 Announce Type: new  Abstract: Regularized nonnegative low-rank approximations such as sparse Nonnegative Matrix Factorization or sparse Nonnegative Tucker Decomposition are an important branch of dimensionality reduction models with enhanced interpretability. However, from a practical perspective, the choice of regularizers and regularization coefficients, as well as the design of efficient algorithms, is challenging because of the multifactor nature of these models and the lack of theory to back these choices. This paper aims at improving upon these issues. By studying a more general model called the Homogeneous Regularized Scale-Invariant, we prove that the scale-invariance inherent to low-rank approximation models causes an implicit regularization with both unexpected beneficial and detrimental effects. This observation allows to better understand the effect of regularization functions in low-rank approximation models, to guide the choice of the regularization hyp
    
[^2]: 基于生成对抗网络的简化建模方法

    Generative Adversarial Reduced Order Modelling. (arXiv:2305.15881v1 [cs.LG])

    [http://arxiv.org/abs/2305.15881](http://arxiv.org/abs/2305.15881)

    本文提出了一种基于GAN的简化建模方法GAROM，通过引入一个数据驱动的生成对抗模型，能够学习参数微分方程的解，并获得了较好的实验效果。

    

    本文提出了一种新的基于生成对抗网络（GAN）的简化建模方法——GAROM。GAN在多个深度学习领域得到广泛应用，但在简化建模中的应用却鲜有研究。我们将GAN和ROM框架相结合，引入了一种数据驱动的生成对抗模型，能够学习参数微分方程的解。我们将鉴别器网络建模为自编码器，提取输入的相关特征，并将微分方程参数作为生成器和鉴别器网络的输入条件。我们展示了如何将该方法应用于推断问题，提供了实验证据证明了模型的泛化能力，并进行了方法的收敛性研究。

    In this work, we present GAROM, a new approach for reduced order modelling (ROM) based on generative adversarial networks (GANs). GANs have the potential to learn data distribution and generate more realistic data. While widely applied in many areas of deep learning, little research is done on their application for ROM, i.e. approximating a high-fidelity model with a simpler one. In this work, we combine the GAN and ROM framework, by introducing a data-driven generative adversarial model able to learn solutions to parametric differential equations. The latter is achieved by modelling the discriminator network as an autoencoder, extracting relevant features of the input, and applying a conditioning mechanism to the generator and discriminator networks specifying the differential equation parameters. We show how to apply our methodology for inference, provide experimental evidence of the model generalisation, and perform a convergence study of the method.
    

