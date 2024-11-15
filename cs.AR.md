# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Embedding Hardware Approximations in Discrete Genetic-based Training for Printed MLPs](https://arxiv.org/abs/2402.02930) | 本文将硬件近似嵌入到印刷多层感知器的训练过程中，通过离散遗传算法实现了最大化硬件近似的效益，在5%的精度损失下，相比基线，实现了超过5倍的面积和功耗的减少，并且超过了最先进的近似方法。 |
| [^2] | [Bespoke Approximation of Multiplication-Accumulation and Activation Targeting Printed Multilayer Perceptrons](https://arxiv.org/abs/2312.17612) | 本研究提出了一种针对印刷电子技术中的限制的自动化框架，用于设计超低功耗的多层感知机（MLP）分类器。 |

# 详细

[^1]: 将硬件近似嵌入离散基因训练中以用于印刷多层感知器

    Embedding Hardware Approximations in Discrete Genetic-based Training for Printed MLPs

    [https://arxiv.org/abs/2402.02930](https://arxiv.org/abs/2402.02930)

    本文将硬件近似嵌入到印刷多层感知器的训练过程中，通过离散遗传算法实现了最大化硬件近似的效益，在5%的精度损失下，相比基线，实现了超过5倍的面积和功耗的减少，并且超过了最先进的近似方法。

    

    印刷电子是一种有着低成本和灵活制造等独特特点的有望广泛应用于计算领域的技术。与传统的硅基技术不同，印刷电子可以实现可伸缩、可适应、非毒性的硬件。然而，由于印刷电子的特性尺寸较大，要实现复杂的电路如机器学习分类器是具有挑战性的。近似计算被证明可以降低机器学习电路（如多层感知器）的硬件成本。在本文中，我们通过将硬件近似嵌入到多层感知器的训练过程中来最大化近似计算的益处。由于硬件近似的离散性，我们提出并实现了一种基于遗传算法的硬件感知训练方法，专门为印刷多层感知器设计。在5%的精度损失下，相比基线，我们的多层感知器在面积和功耗上实现了超过5倍的减少，并且超过了最先进的近似方法。

    Printed Electronics (PE) stands out as a promisingtechnology for widespread computing due to its distinct attributes, such as low costs and flexible manufacturing. Unlike traditional silicon-based technologies, PE enables stretchable, conformal,and non-toxic hardware. However, PE are constrained by larger feature sizes, making it challenging to implement complex circuits such as machine learning (ML) classifiers. Approximate computing has been proven to reduce the hardware cost of ML circuits such as Multilayer Perceptrons (MLPs). In this paper, we maximize the benefits of approximate computing by integrating hardware approximation into the MLP training process. Due to the discrete nature of hardware approximation, we propose and implement a genetic-based, approximate, hardware-aware training approach specifically designed for printed MLPs. For a 5% accuracy loss, our MLPs achieve over 5x area and power reduction compared to the baseline while outperforming state of-the-art approximate
    
[^2]: 面向印刷多层感知机的定制近似乘积累加和激活技术

    Bespoke Approximation of Multiplication-Accumulation and Activation Targeting Printed Multilayer Perceptrons

    [https://arxiv.org/abs/2312.17612](https://arxiv.org/abs/2312.17612)

    本研究提出了一种针对印刷电子技术中的限制的自动化框架，用于设计超低功耗的多层感知机（MLP）分类器。

    

    印刷电子技术具有独特的特性，使其成为实现真正无处不在计算的重要技术。本研究提出了一种自动化框架，用于设计超低功耗的多层感知机（MLP）分类器，通过利用近似计算和定制化设计的原则来克服印刷电子技术中的限制。

    Printed Electronics (PE) feature distinct and remarkable characteristics that make them a prominent technology for achieving true ubiquitous computing. This is particularly relevant in application domains that require conformal and ultra-low cost solutions, which have experienced limited penetration of computing until now. Unlike silicon-based technologies, PE offer unparalleled features such as non-recurring engineering costs, ultra-low manufacturing cost, and on-demand fabrication of conformal, flexible, non-toxic, and stretchable hardware. However, PE face certain limitations due to their large feature sizes, that impede the realization of complex circuits, such as machine learning classifiers. In this work, we address these limitations by leveraging the principles of Approximate Computing and Bespoke (fully-customized) design. We propose an automated framework for designing ultra-low power Multilayer Perceptron (MLP) classifiers which employs, for the first time, a holistic approac
    

