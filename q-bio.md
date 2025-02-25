# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FABind+: Enhancing Molecular Docking through Improved Pocket Prediction and Pose Generation](https://arxiv.org/abs/2403.20261) | FABind+通过改进口袋预测和姿态生成，提升分子对接表现 |
| [^2] | [Bidirectional Graph GAN: Representing Brain Structure-Function Connections for Alzheimer's Disease.](http://arxiv.org/abs/2309.08916) | 本研究提出了一种双向图生成对抗网络（BGGAN），用于表示阿尔茨海默病（AD）的脑结构-功能连接。通过特殊设计的内部图卷积网络模块和平衡器模块，该方法能够准确地学习结构域和功能域之间的映射函数，并解决模式坍塌问题，同时学习结构和功能特征的互补性。 |
| [^3] | [MIML: Multiplex Image Machine Learning for High Precision Cell Classification via Mechanical Traits within Microfluidic Systems.](http://arxiv.org/abs/2309.08421) | 本研究开发了一种新颖的机器学习框架MIML，该框架通过将无标记细胞图像与生物力学属性数据相结合，实现了高精度细胞分类。该方法利用了形态信息，将细胞属性理解得更全面，相较于仅考虑单一数据类型的模型，实现了98.3％的分类精度。该方法已在白细胞和肿瘤细胞分类中得到证明，并具有更广泛的应用潜力。 |

# 详细

[^1]: FABind+: 通过改进口袋预测和姿态生成增强分子对接

    FABind+: Enhancing Molecular Docking through Improved Pocket Prediction and Pose Generation

    [https://arxiv.org/abs/2403.20261](https://arxiv.org/abs/2403.20261)

    FABind+通过改进口袋预测和姿态生成，提升分子对接表现

    

    分子对接是药物发现中至关重要的过程。传统技术依赖于受物理原理支配的广泛采样和模拟，但这些方法往往速度慢且昂贵。基于深度学习的方法的出现显示出显著的前景，提供了精确性和效率的增长。建立在FABind的基础工作之上，这是一个专注于速度和准确性的模型，我们提出了FABind+，这是一个大大提升其前身性能的增强版。我们确定口袋预测是分子对接中的一个关键瓶颈，并提出了一种显著改进口袋预测的新方法，从而简化了对接过程。此外，我们对对接模块进行了修改，以增强其姿态生成能力。为了缩小与传统采样/生成方法之间的差距，我们结合了一个简单而有效的s

    arXiv:2403.20261v1 Announce Type: cross  Abstract: Molecular docking is a pivotal process in drug discovery. While traditional techniques rely on extensive sampling and simulation governed by physical principles, these methods are often slow and costly. The advent of deep learning-based approaches has shown significant promise, offering increases in both accuracy and efficiency. Building upon the foundational work of FABind, a model designed with a focus on speed and accuracy, we present FABind+, an enhanced iteration that largely boosts the performance of its predecessor. We identify pocket prediction as a critical bottleneck in molecular docking and propose a novel methodology that significantly refines pocket prediction, thereby streamlining the docking process. Furthermore, we introduce modifications to the docking module to enhance its pose generation capabilities. In an effort to bridge the gap with conventional sampling/generative methods, we incorporate a simple yet effective s
    
[^2]: 双向图生成对抗网络：用于阿尔茨海默病的脑结构-功能连接的表示

    Bidirectional Graph GAN: Representing Brain Structure-Function Connections for Alzheimer's Disease. (arXiv:2309.08916v1 [cs.AI])

    [http://arxiv.org/abs/2309.08916](http://arxiv.org/abs/2309.08916)

    本研究提出了一种双向图生成对抗网络（BGGAN），用于表示阿尔茨海默病（AD）的脑结构-功能连接。通过特殊设计的内部图卷积网络模块和平衡器模块，该方法能够准确地学习结构域和功能域之间的映射函数，并解决模式坍塌问题，同时学习结构和功能特征的互补性。

    

    揭示脑疾病的发病机制，包括阿尔茨海默病（AD），脑结构与功能之间的关系至关重要。然而，由于各种原因，将脑结构-功能连接映射是一个巨大的挑战。本文提出了一种双向图生成对抗网络（BGGAN）来表示脑结构-功能连接。具体来说，通过设计一个内部图卷积网络（InnerGCN）模块，BGGAN的生成器可以利用直接和间接脑区域的特征来学习结构域和功能域之间的映射函数。此外，还设计了一个名为Balancer的新模块来平衡生成器和判别器之间的优化。通过将Balancer引入到BGGAN中，结构生成器和功能生成器不仅可以缓解模式坍塌问题，还可以学习结构和功能特征的互补性。实验结果表明该方法能够在AD中准确地表示脑结构-功能连接。

    The relationship between brain structure and function is critical for revealing the pathogenesis of brain disease, including Alzheimer's disease (AD). However, it is a great challenge to map brain structure-function connections due to various reasons. In this work, a bidirectional graph generative adversarial networks (BGGAN) is proposed to represent brain structure-function connections. Specifically, by designing a module incorporating inner graph convolution network (InnerGCN), the generators of BGGAN can employ features of direct and indirect brain regions to learn the mapping function between structural domain and functional domain. Besides, a new module named Balancer is designed to counterpoise the optimization between generators and discriminators. By introducing the Balancer into BGGAN, both the structural generator and functional generator can not only alleviate the issue of mode collapse but also learn complementarity of structural and functional features. Experimental result
    
[^3]: MIML: 通过微流控系统内的机械特性对高精度细胞分类进行多重图像机器学习

    MIML: Multiplex Image Machine Learning for High Precision Cell Classification via Mechanical Traits within Microfluidic Systems. (arXiv:2309.08421v1 [eess.IV])

    [http://arxiv.org/abs/2309.08421](http://arxiv.org/abs/2309.08421)

    本研究开发了一种新颖的机器学习框架MIML，该框架通过将无标记细胞图像与生物力学属性数据相结合，实现了高精度细胞分类。该方法利用了形态信息，将细胞属性理解得更全面，相较于仅考虑单一数据类型的模型，实现了98.3％的分类精度。该方法已在白细胞和肿瘤细胞分类中得到证明，并具有更广泛的应用潜力。

    

    无标记细胞分类有助于为进一步使用或检查提供原始细胞，然而现有技术在特异性和速度方面往往不足。在本研究中，我们通过开发一种新颖的机器学习框架MIML来解决这些局限性。该架构将无标记细胞图像与生物力学属性数据相结合，利用每个细胞固有的广阔且常常被低估的形态信息。通过整合这两种类型的数据，我们的模型提供了对细胞属性更全面的理解，利用了传统机器学习模型中通常被丢弃的形态信息。这种方法使细胞分类精度达到了惊人的98.3％，大大优于仅考虑单一数据类型的模型。MIML已被证明在白细胞和肿瘤细胞分类中有效，并具有更广泛的应用潜力。

    Label-free cell classification is advantageous for supplying pristine cells for further use or examination, yet existing techniques frequently fall short in terms of specificity and speed. In this study, we address these limitations through the development of a novel machine learning framework, Multiplex Image Machine Learning (MIML). This architecture uniquely combines label-free cell images with biomechanical property data, harnessing the vast, often underutilized morphological information intrinsic to each cell. By integrating both types of data, our model offers a more holistic understanding of the cellular properties, utilizing morphological information typically discarded in traditional machine learning models. This approach has led to a remarkable 98.3\% accuracy in cell classification, a substantial improvement over models that only consider a single data type. MIML has been proven effective in classifying white blood cells and tumor cells, with potential for broader applicatio
    

