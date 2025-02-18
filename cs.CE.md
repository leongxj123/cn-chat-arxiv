# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Physics-Informed Diffusion Models](https://arxiv.org/abs/2403.14404) | 提出了一个信息化去噪扩散模型框架，可在模型训练期间对生成样本施加约束，以改善样本与约束的对齐程度并提供自然的正则化，适用性广泛。 |
| [^2] | [Topology optimization with physics-informed neural networks: application to noninvasive detection of hidden geometries.](http://arxiv.org/abs/2303.09280) | 该论文介绍了一种基于物理知识神经网络的拓扑优化方法，应用于无先验知识的几何结构检测，通过材料密度场表示任意解决方案拓扑，并通过Eikonal正则化实现。该方法可用于医疗和工业应用中的非侵入式成像技术。 |

# 详细

[^1]: 物理信息扩散模型

    Physics-Informed Diffusion Models

    [https://arxiv.org/abs/2403.14404](https://arxiv.org/abs/2403.14404)

    提出了一个信息化去噪扩散模型框架，可在模型训练期间对生成样本施加约束，以改善样本与约束的对齐程度并提供自然的正则化，适用性广泛。

    

    生成模型如去噪扩散模型正快速提升其逼近高度复杂数据分布的能力。它们也越来越多地被运用于科学机器学习中，预期从隐含数据分布中取样的样本将遵守特定的控制方程。我们提出了一个框架，用于在模型训练期间对生成样本的基础约束进行信息化。我们的方法改善了生成样本与施加约束的对齐程度，显著优于现有方法而不影响推理速度。此外，我们的研究结果表明，在训练过程中加入这些约束提供了自然的防止过拟合的正则化。我们的框架易于实现，适用性广泛，可用于施加等式和不等式约束以及辅助优化目标。

    arXiv:2403.14404v1 Announce Type: new  Abstract: Generative models such as denoising diffusion models are quickly advancing their ability to approximate highly complex data distributions. They are also increasingly leveraged in scientific machine learning, where samples from the implied data distribution are expected to adhere to specific governing equations. We present a framework to inform denoising diffusion models on underlying constraints on such generated samples during model training. Our approach improves the alignment of the generated samples with the imposed constraints and significantly outperforms existing methods without affecting inference speed. Additionally, our findings suggest that incorporating such constraints during training provides a natural regularization against overfitting. Our framework is easy to implement and versatile in its applicability for imposing equality and inequality constraints as well as auxiliary optimization objectives.
    
[^2]: 物理知识神经网络拓扑优化：应用于隐藏几何结构的非侵入式探测。

    Topology optimization with physics-informed neural networks: application to noninvasive detection of hidden geometries. (arXiv:2303.09280v1 [cs.LG])

    [http://arxiv.org/abs/2303.09280](http://arxiv.org/abs/2303.09280)

    该论文介绍了一种基于物理知识神经网络的拓扑优化方法，应用于无先验知识的几何结构检测，通过材料密度场表示任意解决方案拓扑，并通过Eikonal正则化实现。该方法可用于医疗和工业应用中的非侵入式成像技术。

    

    在医疗和工业应用中，通过电磁、声学或机械负载从表面测量中检测隐藏的几何结构是非侵入成像技术的目标。由于未知的拓扑和几何形状、数据的稀疏性以及物理规律的复杂性，解决逆问题是具有挑战性的。物理知识神经网络已经表现出许多优点，是一个简单而强大的问题反演工具，但它们尚未应用于具有先验未知拓扑的一般问题。在这里，我们介绍了一个基于PINNs的拓扑优化框架，它可以解决没有形状数量或类型先验知识的几何检测问题。我们允许任意的解决方案拓扑，通过使用材料密度场来表示几何形状，并通过新的Eikonal正则化接近二进制值。我们通过检测隐含虚空和包含物的数量、位置和形状来验证我们的框架。

    Detecting hidden geometrical structures from surface measurements under electromagnetic, acoustic, or mechanical loading is the goal of noninvasive imaging techniques in medical and industrial applications. Solving the inverse problem can be challenging due to the unknown topology and geometry, the sparsity of the data, and the complexity of the physical laws. Physics-informed neural networks (PINNs) have shown promise as a simple-yet-powerful tool for problem inversion, but they have yet to be applied to general problems with a priori unknown topology. Here, we introduce a topology optimization framework based on PINNs that solves geometry detection problems without prior knowledge of the number or types of shapes. We allow for arbitrary solution topology by representing the geometry using a material density field that approaches binary values thanks to a novel eikonal regularization. We validate our framework by detecting the number, locations, and shapes of hidden voids and inclusio
    

