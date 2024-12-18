# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SceneX:Procedural Controllable Large-scale Scene Generation via Large-language Models](https://arxiv.org/abs/2403.15698) | 通过大型语言模型驱动程序化建模，提出了一个大规模场景生成框架SceneX，可以自动生成高质量的程序化模型 |
| [^2] | [Can Generative Models Improve Self-Supervised Representation Learning?](https://arxiv.org/abs/2403.05966) | 本文引入了一个新的框架，通过利用生成模型生成语义一致的图像增强，丰富了自监督学习的方法，实验结果表明提高了学到的视觉质量。 |
| [^3] | [SemPLeS: Semantic Prompt Learning for Weakly-Supervised Semantic Segmentation](https://arxiv.org/abs/2401.11791) | SemPLeS框架利用语义提示学习解决弱监督语义分割中的问题，通过学习有效提示来增强分割区域与目标对象类别之间的语义对齐。 |
| [^4] | [JEN-1 Composer: A Unified Framework for High-Fidelity Multi-Track Music Generation.](http://arxiv.org/abs/2310.19180) | JEN-1 Composer是一个统一的框架，能够以高保真、灵活的方式生成多音轨音乐。 |
| [^5] | [OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection.](http://arxiv.org/abs/2306.09301) | OpenOOD v1.5 是对前身的重大改进，将OCC检测方法的评估能力扩展到大规模数据集，调查了全光谱OCC检测，引入了在线排行榜和易于使用的评估器等新功能，并提供了深入的分析和实验结果的见解。 |

# 详细

[^1]: SceneX：通过大型语言模型进行程序化可控大规模场景生成

    SceneX:Procedural Controllable Large-scale Scene Generation via Large-language Models

    [https://arxiv.org/abs/2403.15698](https://arxiv.org/abs/2403.15698)

    通过大型语言模型驱动程序化建模，提出了一个大规模场景生成框架SceneX，可以自动生成高质量的程序化模型

    

    由于其巨大的应用潜力，大规模场景生成在学术界和工业界引起了广泛关注。最近的研究采用强大的生成模型创建所需的场景，并取得了令人期待的结果。然而，大多数这些方法使用不兼容工业流程的3D基元（如点云或辐射场）来表示场景，这导致学术研究与工业部署之间存在重大差距。程序化可控生成（PCG）是一种高效的技术，可创建可扩展和高质量的资产，但对普通用户不友好，因为它需要深入的领域专业知识。为解决这些问题，我们采用大型语言模型（LLM）驱动程序化建模。在本文中，我们介绍了一个大规模场景生成框架SceneX，可以根据设计师的文本描述自动生成高质量的程序化模型。

    arXiv:2403.15698v1 Announce Type: cross  Abstract: Due to its great application potential, large-scale scene generation has drawn extensive attention in academia and industry. Recent research employs powerful generative models to create desired scenes and achieves promising results. However, most of these methods represent the scene using 3D primitives (e.g. point cloud or radiance field) incompatible with the industrial pipeline, which leads to a substantial gap between academic research and industrial deployment. Procedural Controllable Generation (PCG) is an efficient technique for creating scalable and high-quality assets, but it is unfriendly for ordinary users as it demands profound domain expertise. To address these issues, we resort to using the large language model (LLM) to drive the procedural modeling. In this paper, we introduce a large-scale scene generation framework, SceneX, which can automatically produce high-quality procedural models according to designers' textual de
    
[^2]: 能生成模型改进自监督表示学习吗？

    Can Generative Models Improve Self-Supervised Representation Learning?

    [https://arxiv.org/abs/2403.05966](https://arxiv.org/abs/2403.05966)

    本文引入了一个新的框架，通过利用生成模型生成语义一致的图像增强，丰富了自监督学习的方法，实验结果表明提高了学到的视觉质量。

    

    自监督学习的快速发展突显了其利用无标签数据学习强大视觉表示的潜力。然而，现有的自监督学习方法，特别是那些利用同一图像的不同视图的方法，通常依赖于一组预定义的数据增强，这限制了变换的多样性和质量，导致表示不够优化。本文引入了一个新颖的框架，通过利用生成模型产生语义一致的图像增强，丰富了自监督学习范式。通过直接在源图像表示上进行条件生成模型，我们的方法能够生成多样的增强，同时保持源图像的语义，为自监督学习提供更丰富的数据集。我们的实验结果表明，我们的框架显著提高了学到的视觉的质量。

    arXiv:2403.05966v1 Announce Type: cross  Abstract: The rapid advancement in self-supervised learning (SSL) has highlighted its potential to leverage unlabeled data for learning powerful visual representations. However, existing SSL approaches, particularly those employing different views of the same image, often rely on a limited set of predefined data augmentations. This constrains the diversity and quality of transformations, which leads to sub-optimal representations. In this paper, we introduce a novel framework that enriches the SSL paradigm by utilizing generative models to produce semantically consistent image augmentations. By directly conditioning generative models on a source image representation, our method enables the generation of diverse augmentations while maintaining the semantics of the source image, thus offering a richer set of data for self-supervised learning. Our experimental results demonstrate that our framework significantly enhances the quality of learned visu
    
[^3]: SemPLeS: 语义提示学习用于弱监督语义分割

    SemPLeS: Semantic Prompt Learning for Weakly-Supervised Semantic Segmentation

    [https://arxiv.org/abs/2401.11791](https://arxiv.org/abs/2401.11791)

    SemPLeS框架利用语义提示学习解决弱监督语义分割中的问题，通过学习有效提示来增强分割区域与目标对象类别之间的语义对齐。

    

    弱监督语义分割（WSSS）旨在利用仅具有图像级监督的图像数据来训练分割模型。由于无法获得精确的像素级标注，现有方法通常侧重于通过优化CAM样式的热图来生成用于训练分割模型的伪标记。然而，生成的热图可能仅捕获对象类别的具有区分性的图像区域或相关的共同出现的背景。为解决这些问题，我们提出了一种用于WSSS的语义提示学习（SemPLeS）框架，该框架学习有效地提示CLIP潜空间以增强分割区域与目标对象类别之间的语义对准。具体而言，我们提出了对比提示学习和提示引导的语义细化，以学习适当描述和抑制与每个目标对象类别相关的共同出现的背景的提示。

    arXiv:2401.11791v2 Announce Type: replace-cross  Abstract: Weakly-Supervised Semantic Segmentation (WSSS) aims to train segmentation models using image data with only image-level supervision. Since precise pixel-level annotations are not accessible, existing methods typically focus on producing pseudo masks for training segmentation models by refining CAM-like heatmaps. However, the produced heatmaps may capture only the discriminative image regions of object categories or the associated co-occurring backgrounds. To address the issues, we propose a Semantic Prompt Learning for WSSS (SemPLeS) framework, which learns to effectively prompt the CLIP latent space to enhance the semantic alignment between the segmented regions and the target object categories. More specifically, we propose Contrastive Prompt Learning and Prompt-guided Semantic Refinement to learn the prompts that adequately describe and suppress the co-occurring backgrounds associated with each target object category. In thi
    
[^4]: JEN-1 Composer: 一个用于高保真多音轨音乐生成的统一框架

    JEN-1 Composer: A Unified Framework for High-Fidelity Multi-Track Music Generation. (arXiv:2310.19180v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2310.19180](http://arxiv.org/abs/2310.19180)

    JEN-1 Composer是一个统一的框架，能够以高保真、灵活的方式生成多音轨音乐。

    

    随着生成式人工智能的快速发展，从零开始生成音乐的文本到音乐合成任务已成为一个有前景的方向。然而，对于多音轨生成的更细粒度控制仍然是一个挑战。现有模型具有较强的原始生成能力，但缺乏以可控的方式单独组成和组合多音轨的灵活性，这与人类作曲家的典型工作流程不同。为了解决这个问题，我们提出了JEN-1 Composer，一个统一的框架，通过一个模型高效地建模多音轨音乐的边缘、条件和联合分布。JEN-1 Composer框架能够无缝地整合任何基于扩散的音乐生成系统，例如Jen-1，增强其多功能多音轨音乐生成能力。我们引入了一种课程训练策略，以逐步指导模型从单音轨生成到灵活的生成过程。

    With rapid advances in generative artificial intelligence, the text-to-music synthesis task has emerged as a promising direction for music generation from scratch. However, finer-grained control over multi-track generation remains an open challenge. Existing models exhibit strong raw generation capability but lack the flexibility to compose separate tracks and combine them in a controllable manner, differing from typical workflows of human composers. To address this issue, we propose JEN-1 Composer, a unified framework to efficiently model marginal, conditional, and joint distributions over multi-track music via a single model. JEN-1 Composer framework exhibits the capacity to seamlessly incorporate any diffusion-based music generation system, \textit{e.g.} Jen-1, enhancing its capacity for versatile multi-track music generation. We introduce a curriculum training strategy aimed at incrementally instructing the model in the transition from single-track generation to the flexible genera
    
[^5]: OpenOOD v1.5：增强的OCC（Out-of-Distribution Detection）基准测试

    OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection. (arXiv:2306.09301v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.09301](http://arxiv.org/abs/2306.09301)

    OpenOOD v1.5 是对前身的重大改进，将OCC检测方法的评估能力扩展到大规模数据集，调查了全光谱OCC检测，引入了在线排行榜和易于使用的评估器等新功能，并提供了深入的分析和实验结果的见解。

    

    OCC检测对于开放世界智能系统的可靠运行至关重要。虽然出现了越来越多的OCC检测方法，但评估不一致性仍然存在挑战，难以跟踪该领域的进展。本文介绍了OpenOOD v1.5，这是对前身的重大改进，确保OCC检测方法的准确、标准化和用户友好的评估。值得注意的是，OpenOOD v1.5将其评估能力扩展到大规模数据集，如ImageNet。此外，它还调查了全光谱OCC检测，引入了在线排行榜和易于使用的评估器等新功能。该工作还提供了深入的分析和综合实验结果的见解，从而丰富了知识库。

    Out-of-Distribution (OOD) detection is critical for the reliable operation of open-world intelligent systems. Despite the emergence of an increasing number of OOD detection methods, the evaluation inconsistencies present challenges for tracking the progress in this field. OpenOOD v1 initiated the unification of the OOD detection evaluation but faced limitations in scalability and usability. In response, this paper presents OpenOOD v1.5, a significant improvement from its predecessor that ensures accurate, standardized, and user-friendly evaluation of OOD detection methodologies. Notably, OpenOOD v1.5 extends its evaluation capabilities to large-scale datasets such as ImageNet, investigates full-spectrum OOD detection which is important yet underexplored, and introduces new features including an online leaderboard and an easy-to-use evaluator. This work also contributes in-depth analysis and insights derived from comprehensive experimental results, thereby enriching the knowledge pool o
    

