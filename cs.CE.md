# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Physics-Informed Diffusion Models](https://arxiv.org/abs/2403.14404) | 提出了一个信息化去噪扩散模型框架，可在模型训练期间对生成样本施加约束，以改善样本与约束的对齐程度并提供自然的正则化，适用性广泛。 |

# 详细

[^1]: 物理信息扩散模型

    Physics-Informed Diffusion Models

    [https://arxiv.org/abs/2403.14404](https://arxiv.org/abs/2403.14404)

    提出了一个信息化去噪扩散模型框架，可在模型训练期间对生成样本施加约束，以改善样本与约束的对齐程度并提供自然的正则化，适用性广泛。

    

    生成模型如去噪扩散模型正快速提升其逼近高度复杂数据分布的能力。它们也越来越多地被运用于科学机器学习中，预期从隐含数据分布中取样的样本将遵守特定的控制方程。我们提出了一个框架，用于在模型训练期间对生成样本的基础约束进行信息化。我们的方法改善了生成样本与施加约束的对齐程度，显著优于现有方法而不影响推理速度。此外，我们的研究结果表明，在训练过程中加入这些约束提供了自然的防止过拟合的正则化。我们的框架易于实现，适用性广泛，可用于施加等式和不等式约束以及辅助优化目标。

    arXiv:2403.14404v1 Announce Type: new  Abstract: Generative models such as denoising diffusion models are quickly advancing their ability to approximate highly complex data distributions. They are also increasingly leveraged in scientific machine learning, where samples from the implied data distribution are expected to adhere to specific governing equations. We present a framework to inform denoising diffusion models on underlying constraints on such generated samples during model training. Our approach improves the alignment of the generated samples with the imposed constraints and significantly outperforms existing methods without affecting inference speed. Additionally, our findings suggest that incorporating such constraints during training provides a natural regularization against overfitting. Our framework is easy to implement and versatile in its applicability for imposing equality and inequality constraints as well as auxiliary optimization objectives.
    

