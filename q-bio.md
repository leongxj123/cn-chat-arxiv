# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sculpting Molecules in 3D: A Flexible Substructure Aware Framework for Text-Oriented Molecular Optimization](https://arxiv.org/abs/2403.03425) | 提出了一种通过多模态引导生成/优化任务解决分子设计问题的创新方法。 |
| [^2] | [Gode -- Integrating Biochemical Knowledge Graph into Pre-training Molecule Graph Neural Network.](http://arxiv.org/abs/2306.01631) | 本研究提出了一种新的方法，在分子结构和生物医学知识图谱中集成多个领域信息，通过自我监督策略预先训练更广泛和更强大的表示，并在化学属性预测任务上展示出出色的性能。 |

# 详细

[^1]: 在3D中塑造分子：面向文本的分子优化灵活子结构感知框架

    Sculpting Molecules in 3D: A Flexible Substructure Aware Framework for Text-Oriented Molecular Optimization

    [https://arxiv.org/abs/2403.03425](https://arxiv.org/abs/2403.03425)

    提出了一种通过多模态引导生成/优化任务解决分子设计问题的创新方法。

    

    通过将深度学习，特别是AI-Generated Content，与从第一原理计算中得出的高质量数据相结合，已经成为改变科学研究格局的一种有前途的途径。然而，设计既包含多模态先验知识又具有关键和复杂性的分子药物或材料的挑战依然是一项关键而复杂的工作。本文提出了一种创新方法来解决这一逆设计问题，将其构造为一种多模态导向生成/优化任务。我们提出的解决方案涉及一个面向文本-结构对齐的对称扩散框架，用于实现分子生成/优化任务，即3DToMolo.

    arXiv:2403.03425v1 Announce Type: new  Abstract: The integration of deep learning, particularly AI-Generated Content, with high-quality data derived from ab initio calculations has emerged as a promising avenue for transforming the landscape of scientific research. However, the challenge of designing molecular drugs or materials that incorporate multi-modality prior knowledge remains a critical and complex undertaking. Specifically, achieving a practical molecular design necessitates not only meeting the diversity requirements but also addressing structural and textural constraints with various symmetries outlined by domain experts. In this article, we present an innovative approach to tackle this inverse design problem by formulating it as a multi-modality guidance generation/optimization task. Our proposed solution involves a textural-structure alignment symmetric diffusion framework for the implementation of molecular generation/optimization tasks, namely 3DToMolo. 3DToMolo aims to 
    
[^2]: Gode -- 将生物化学知识图谱集成到分子图神经网络的预训练中

    Gode -- Integrating Biochemical Knowledge Graph into Pre-training Molecule Graph Neural Network. (arXiv:2306.01631v1 [cs.LG])

    [http://arxiv.org/abs/2306.01631](http://arxiv.org/abs/2306.01631)

    本研究提出了一种新的方法，在分子结构和生物医学知识图谱中集成多个领域信息，通过自我监督策略预先训练更广泛和更强大的表示，并在化学属性预测任务上展示出出色的性能。

    

    分子属性的准确预测对于促进创新治疗方法的发展和理解化学物质和生物系统之间复杂的相互作用至关重要。本研究提出了一种新的方法，将单个分子结构的图表示与生物医学知识图谱 (KG) 的多个领域信息进行集成。通过集成两个级别的信息，我们可以使用自我监督策略预先训练更广泛和更强大的表示，用于分子级和 KG 级预测任务。在性能评估方面，我们在 11 个具有挑战性的化学属性预测任务上微调我们预先训练的模型。我们的框架的结果表明，我们微调的模型优于现有的最先进的模型。

    The precise prediction of molecular properties holds paramount importance in facilitating the development of innovative treatments and comprehending the intricate interplay between chemicals and biological systems. In this study, we propose a novel approach that integrates graph representations of individual molecular structures with multi-domain information from biomedical knowledge graphs (KGs). Integrating information from both levels, we can pre-train a more extensive and robust representation for both molecule-level and KG-level prediction tasks with our novel self-supervision strategy. For performance evaluation, we fine-tune our pre-trained model on 11 challenging chemical property prediction tasks. Results from our framework demonstrate our fine-tuned models outperform existing state-of-the-art models.
    

