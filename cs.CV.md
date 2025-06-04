# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PointCloud-Text Matching: Benchmark Datasets and a Baseline](https://arxiv.org/abs/2403.19386) | 本文提出了一个新的实例级检索任务：PointCloud-Text匹配（PTM），并构建了三个新的基准数据集以解决数据稀疏、文本模糊等挑战，同时提出了RoMa方法作为PTM的基线模型。 |
| [^2] | [T-TAME: Trainable Attention Mechanism for Explaining Convolutional Networks and Vision Transformers](https://arxiv.org/abs/2403.04523) | 本文提出了T-TAME，一种适用于卷积网络和视觉Transformer的可训练注意机制，为解释深度神经网络在图像分类任务中的应用提供了通用方法。 |
| [^3] | [Open-world Machine Learning: A Review and New Outlooks](https://arxiv.org/abs/2403.01759) | 通过研究未知拒绝、新类别发现和类别增量学习，本文拓展了开放世界机器学习领域，提出了未来研究的多个潜在方向 |
| [^4] | [The Male CEO and the Female Assistant: Probing Gender Biases in Text-To-Image Models Through Paired Stereotype Test](https://arxiv.org/abs/2402.11089) | 通过成对刻板印象测试（PST）框架，在文本-图像模型中探究性别偏见，并评估了DALLE-3在性别职业和组织权力方面的偏见。 |

# 详细

[^1]: PointCloud-Text匹配：基准数据集和一个基线

    PointCloud-Text Matching: Benchmark Datasets and a Baseline

    [https://arxiv.org/abs/2403.19386](https://arxiv.org/abs/2403.19386)

    本文提出了一个新的实例级检索任务：PointCloud-Text匹配（PTM），并构建了三个新的基准数据集以解决数据稀疏、文本模糊等挑战，同时提出了RoMa方法作为PTM的基线模型。

    

    在本文中，我们介绍和研究了一个新的实例级检索任务：PointCloud-Text Matching（PTM），旨在找到与给定的点云查询或文本查询匹配的确切跨模态实例。PTM可应用于各种场景，如室内/城市峡谷定位和场景检索。然而，在实践中尚无适用的、有针对性的PTM数据集。因此，我们构建了三个新的PTM基准数据集，分别为3D2T-SR、3D2T-NR和3D2T-QA。我们观察到数据具有挑战性，由于点云的稀疏、噪声或无序，以及文本的模糊、含糊或不完整，导致存在嘈杂的对应关系，使得现有的跨模态匹配方法对PTM无效。为了解决这些挑战，我们提出了一个PTM基线，命名为Robust PointCloud-Text Matching方法（RoMa）。RoMa包含两个模块：双重注意感知模块（DAP）和鲁棒负对比模块

    arXiv:2403.19386v1 Announce Type: cross  Abstract: In this paper, we present and study a new instance-level retrieval task: PointCloud-Text Matching~(PTM), which aims to find the exact cross-modal instance that matches a given point-cloud query or text query. PTM could be applied to various scenarios, such as indoor/urban-canyon localization and scene retrieval. However, there exists no suitable and targeted dataset for PTM in practice. Therefore, we construct three new PTM benchmark datasets, namely 3D2T-SR, 3D2T-NR, and 3D2T-QA. We observe that the data is challenging and with noisy correspondence due to the sparsity, noise, or disorder of point clouds and the ambiguity, vagueness, or incompleteness of texts, which make existing cross-modal matching methods ineffective for PTM. To tackle these challenges, we propose a PTM baseline, named Robust PointCloud-Text Matching method (RoMa). RoMa consists of two modules: a Dual Attention Perception module (DAP) and a Robust Negative Contrast
    
[^2]: T-TAME：用于解释卷积网络和视觉Transformer的可训练注意机制

    T-TAME: Trainable Attention Mechanism for Explaining Convolutional Networks and Vision Transformers

    [https://arxiv.org/abs/2403.04523](https://arxiv.org/abs/2403.04523)

    本文提出了T-TAME，一种适用于卷积网络和视觉Transformer的可训练注意机制，为解释深度神经网络在图像分类任务中的应用提供了通用方法。

    

    Vision Transformers和其他用于图像分类任务的深度学习架构的发展和应用快速增长。然而，神经网络的“黑匣子”特性是在需要解释性的应用中采用的障碍。虽然已经提出了一些用于生成解释的技术，主要用于卷积神经网络，但是将这些技术适应到视觉Transformer的新范式是非平凡的。本文提出了T-TAME，Transformer兼容的可训练注意机制用于解释，这是一种说明用于图像分类任务中的深度神经网络的通用方法。所提出的架构和训练技术可以轻松应用于任何卷积或类似Vision Transformer的神经网络，使用精简的训练方法。训练后，解释图可以在单次前向传递中计算出；这些解释图可以与Convolutional Neural Networks中生成的解释图相媲美或者

    arXiv:2403.04523v1 Announce Type: cross  Abstract: The development and adoption of Vision Transformers and other deep-learning architectures for image classification tasks has been rapid. However, the "black box" nature of neural networks is a barrier to adoption in applications where explainability is essential. While some techniques for generating explanations have been proposed, primarily for Convolutional Neural Networks, adapting such techniques to the new paradigm of Vision Transformers is non-trivial. This paper presents T-TAME, Transformer-compatible Trainable Attention Mechanism for Explanations, a general methodology for explaining deep neural networks used in image classification tasks. The proposed architecture and training technique can be easily applied to any convolutional or Vision Transformer-like neural network, using a streamlined training approach. After training, explanation maps can be computed in a single forward pass; these explanation maps are comparable to or 
    
[^3]: 开放世界机器学习：回顾与新展望

    Open-world Machine Learning: A Review and New Outlooks

    [https://arxiv.org/abs/2403.01759](https://arxiv.org/abs/2403.01759)

    通过研究未知拒绝、新类别发现和类别增量学习，本文拓展了开放世界机器学习领域，提出了未来研究的多个潜在方向

    

    机器学习在许多应用中取得了显著成功。然而，现有研究主要基于封闭世界假设，即假定环境是静态的，模型一旦部署就是固定的。在许多现实应用中，这种基本且相当幼稚的假设可能不成立，因为开放环境复杂、动态且充满未知。在这种情况下，拒绝未知、发现新奇点，然后逐步学习，可以使模型像生物系统一样安全地并持续进化。本文通过研究未知拒绝、新类别发现和类别增量学习在统一范式中，提供了对开放世界机器学习的整体观点。详细讨论了当前方法的挑战、原则和局限性。最后，我们讨论了几个未来研究的潜在方向。本文旨在提供一份综述

    arXiv:2403.01759v1 Announce Type: new  Abstract: Machine learning has achieved remarkable success in many applications. However, existing studies are largely based on the closed-world assumption, which assumes that the environment is stationary, and the model is fixed once deployed. In many real-world applications, this fundamental and rather naive assumption may not hold because an open environment is complex, dynamic, and full of unknowns. In such cases, rejecting unknowns, discovering novelties, and then incrementally learning them, could enable models to be safe and evolve continually as biological systems do. This paper provides a holistic view of open-world machine learning by investigating unknown rejection, novel class discovery, and class-incremental learning in a unified paradigm. The challenges, principles, and limitations of current methodologies are discussed in detail. Finally, we discuss several potential directions for future research. This paper aims to provide a compr
    
[^4]: 男性CEO和女性助理：通过成对刻板印象测试探究文本-图像模型中的性别偏见

    The Male CEO and the Female Assistant: Probing Gender Biases in Text-To-Image Models Through Paired Stereotype Test

    [https://arxiv.org/abs/2402.11089](https://arxiv.org/abs/2402.11089)

    通过成对刻板印象测试（PST）框架，在文本-图像模型中探究性别偏见，并评估了DALLE-3在性别职业和组织权力方面的偏见。

    

    最近大规模的文本到图像（T2I）模型（如DALLE-3）展示了在新应用中的巨大潜力，但也面临前所未有的公平挑战。先前的研究揭示了单人图像生成中的性别偏见，但T2I模型应用可能需要同时描绘两个或更多人。该设定中的潜在偏见仍未被探究，导致使用中的公平相关风险。为了研究T2I模型中性别偏见的基本方面，我们提出了一种新颖的成对刻板印象测试（PST）偏见评估框架。PST促使模型生成同一图像中的两个个体，用与相反性别刻板印象相关联的两个社会身份来描述他们。通过生成的图像遵从性别刻板印象的程度来衡量偏见。利用PST，我们从两个角度评估DALLE-3：性别职业中的偏见和组织权力中的偏见。

    arXiv:2402.11089v1 Announce Type: cross  Abstract: Recent large-scale Text-To-Image (T2I) models such as DALLE-3 demonstrate great potential in new applications, but also face unprecedented fairness challenges. Prior studies revealed gender biases in single-person image generation, but T2I model applications might require portraying two or more people simultaneously. Potential biases in this setting remain unexplored, leading to fairness-related risks in usage. To study these underlying facets of gender biases in T2I models, we propose a novel Paired Stereotype Test (PST) bias evaluation framework. PST prompts the model to generate two individuals in the same image. They are described with two social identities that are stereotypically associated with the opposite gender. Biases can then be measured by the level of conformation to gender stereotypes in generated images. Using PST, we evaluate DALLE-3 from 2 perspectives: biases in gendered occupation and biases in organizational power.
    

