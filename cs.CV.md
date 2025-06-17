# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Counterfactual contrastive learning: robust representations via causal image synthesis](https://arxiv.org/abs/2403.09605) | 本研究提出了CF-SimCLR，一种反事实对照学习方法，利用近似反事实推断创造正样本，大大提高了模型对采集偏移的稳健性，并在多个数据集上取得了较高的下游性能。 |
| [^2] | [Video Understanding with Large Language Models: A Survey.](http://arxiv.org/abs/2312.17432) | 这项调查研究提供了对大型语言模型（Vid-LLMs）在视频理解中的最新进展的详细概述。Vid-LLMs的新兴能力包括开放式时空推理和常识知识，为未来的视频理解提供了有前途的方向。 |
| [^3] | [CLCIFAR: CIFAR-Derived Benchmark Datasets with Human Annotated Complementary Labels.](http://arxiv.org/abs/2305.08295) | 本研究开发了由人类标注的互补标签，创造了两个真实世界的CLL数据集，进一步揭示了现实表现下CLL算法的性能，为这一领域的研究提供了更实际的评估标准。 |
| [^4] | [Geometry-Aware Latent Representation Learning for Modeling Disease Progression of Barrett's Esophagus.](http://arxiv.org/abs/2303.12711) | 本文提出了一种基于几何思想的潜在表示学习方法，用于建模Barrett食管疾病进程，与传统方法相比，具有更好的重建损失。 |
| [^5] | [Efficient Multi-order Gated Aggregation Network.](http://arxiv.org/abs/2211.03295) | 本文探索了现代卷积神经网络的表征能力，使用多阶博弈论交互的新视角，提出了一种新的纯卷积神经网络架构MogaNet，它表现出优异的可扩展性，并在多种典型视觉基准中以更高效的参数利用达到了与最先进模型竞争的效果。 |

# 详细

[^1]: 反事实对照学习：通过因果图像合成获得稳健表示

    Counterfactual contrastive learning: robust representations via causal image synthesis

    [https://arxiv.org/abs/2403.09605](https://arxiv.org/abs/2403.09605)

    本研究提出了CF-SimCLR，一种反事实对照学习方法，利用近似反事实推断创造正样本，大大提高了模型对采集偏移的稳健性，并在多个数据集上取得了较高的下游性能。

    

    对比预训练已被广泛认为能够提高下游任务性能和模型泛化能力，特别是在有限标签设置中。然而，它对增强管道的选择敏感。正样本应保留语义信息同时破坏域特定信息。标准增强管道通过预定义的光度变换模拟域特定变化，但如果我们能够模拟真实的领域变化呢？在这项工作中，我们展示了如何利用最近在反事实图像生成方面的进展来实现这一目的。我们提出了CF-SimCLR，一种反事实对照学习方法，它利用近似反事实推断进行正样本创建。对胸部X光和乳腺X光等五个数据集的全面评估表明，CF-SimCLR显著提高了对获取偏移的稳健性，在两种数据集上的下游性能更好。

    arXiv:2403.09605v1 Announce Type: cross  Abstract: Contrastive pretraining is well-known to improve downstream task performance and model generalisation, especially in limited label settings. However, it is sensitive to the choice of augmentation pipeline. Positive pairs should preserve semantic information while destroying domain-specific information. Standard augmentation pipelines emulate domain-specific changes with pre-defined photometric transformations, but what if we could simulate realistic domain changes instead? In this work, we show how to utilise recent progress in counterfactual image generation to this effect. We propose CF-SimCLR, a counterfactual contrastive learning approach which leverages approximate counterfactual inference for positive pair creation. Comprehensive evaluation across five datasets, on chest radiography and mammography, demonstrates that CF-SimCLR substantially improves robustness to acquisition shift with higher downstream performance on both in- an
    
[^2]: 大型语言模型在视频理解中的应用：一项调查研究

    Video Understanding with Large Language Models: A Survey. (arXiv:2312.17432v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.17432](http://arxiv.org/abs/2312.17432)

    这项调查研究提供了对大型语言模型（Vid-LLMs）在视频理解中的最新进展的详细概述。Vid-LLMs的新兴能力包括开放式时空推理和常识知识，为未来的视频理解提供了有前途的方向。

    

    随着在线视频平台的不断增长和视频内容的不断增多，对熟练的视频理解工具的需求显著增加。鉴于大型语言模型在语言和多模态任务中的卓越能力，本调查提供了对利用大型语言模型（Vid-LLMs）技术进行视频理解的最新进展的详细概述。Vid-LLMs的新兴能力令人惊讶，尤其是它们在开放式时空推理和常识知识方面的能力，为未来的视频理解提供了一个有前途的方向。本调查对Vid-LLMs的独特特点和能力进行了分类，分为四种主要类型：基于LLM的视频代理、Vid-LLMs的预训练、Vid-LLMs的指令调整和混合方法。此外，本调查对Vid-LLMs的任务、数据集和评估方法进行了全面的研究。另外，它还探讨了Vid-LLMs技术的局限性和未来的挑战。

    With the burgeoning growth of online video platforms and the escalating volume of video content, the demand for proficient video understanding tools has intensified markedly. Given the remarkable capabilities of Large Language Models (LLMs) in language and multimodal tasks, this survey provides a detailed overview of the recent advancements in video understanding harnessing the power of LLMs (Vid-LLMs). The emergent capabilities of Vid-LLMs are surprisingly advanced, particularly their ability for open-ended spatial-temporal reasoning combined with commonsense knowledge, suggesting a promising path for future video understanding. We examine the unique characteristics and capabilities of Vid-LLMs, categorizing the approaches into four main types: LLM-based Video Agents, Vid-LLMs Pretraining, Vid-LLMs Instruction Tuning, and Hybrid Methods. Furthermore, this survey presents a comprehensive study of the tasks, datasets, and evaluation methodologies for Vid-LLMs. Additionally, it explores 
    
[^3]: CLCIFAR：带人类标注互补标签的CIFAR派生基准数据集

    CLCIFAR: CIFAR-Derived Benchmark Datasets with Human Annotated Complementary Labels. (arXiv:2305.08295v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.08295](http://arxiv.org/abs/2305.08295)

    本研究开发了由人类标注的互补标签，创造了两个真实世界的CLL数据集，进一步揭示了现实表现下CLL算法的性能，为这一领域的研究提供了更实际的评估标准。

    

    互补标签学习（CLL）是一种弱监督学习范式，旨在仅使用互补标签（标示实例不属于哪些类别）来训练多类分类器。尽管已经提出了多种CLL算法，但由于两个原因，它们的实际表现仍不清楚。首先，这些算法通常依赖于对互补标签生成的假设。其次，它们的评估仅限于合成数据集。为了获取有关CLL算法的真实世界表现的见解，我们开发了一种协议来收集由人类注释者注释的互补标签。这一努力导致创建了两个数据集，CLCIFAR10和CLCIFAR20，分别由CIFAR10和CIFAR100派生而来。这些数据集在https://github.com/ntucllab/complementary_cifar上公开发布，代表了第一个真实世界的CLL数据集。通过广泛的基准实验，我们发现相较于合成数据集，当使用人类注释的互补标签时，性能有明显下降。但是，我们也观察到，真实世界的CLL数据集使得在更接近实际应用条件下评估算法成为可能，从而更真实地评估其性能。

    Complementary-label learning (CLL) is a weakly-supervised learning paradigm that aims to train a multi-class classifier using only complementary labels, which indicate classes to which an instance does not belong. Despite numerous algorithmic proposals for CLL, their practical performance remains unclear for two reasons. Firstly, these algorithms often rely on assumptions about the generation of complementary labels. Secondly, their evaluation has been limited to synthetic datasets. To gain insights into the real-world performance of CLL algorithms, we developed a protocol to collect complementary labels annotated by human annotators. This effort resulted in the creation of two datasets, CLCIFAR10 and CLCIFAR20, derived from CIFAR10 and CIFAR100, respectively. These datasets, publicly released at https://github.com/ntucllab/complementary_cifar, represent the very first real-world CLL datasets. Through extensive benchmark experiments, we discovered a notable decline in performance when 
    
[^4]: 基于几何感知的潜在表示学习用于建模Barrett食管疾病进程

    Geometry-Aware Latent Representation Learning for Modeling Disease Progression of Barrett's Esophagus. (arXiv:2303.12711v1 [eess.IV])

    [http://arxiv.org/abs/2303.12711](http://arxiv.org/abs/2303.12711)

    本文提出了一种基于几何思想的潜在表示学习方法，用于建模Barrett食管疾病进程，与传统方法相比，具有更好的重建损失。

    

    Barrett食管是食管腺癌的唯一先驱，这是一种在诊断时预后不良的食管癌症。因此，诊断Barrett食管对于预防和治疗食管癌至关重要。监督机器学习支持Barrett食管诊断，但组织病理学训练数据的高观察者变异限制了这些方法。用变分自动编码器(VAEs)进行无监督表示学习显示出潜在优势，因为它们将输入数据映射到具有仅有用特征的低维流形，为改进下游任务和见解将Barrett食管病程表征。然而，VAE的欧几里得潜在空间扭曲了点之间的关系，从而阻碍了疾病进展建模。几何VAEs为潜在空间提供附加几何结构，RHVAE假设为黎曼流形，$\mathcal{S}$-VAE假设为超球面流形。我们的研究表明，$\mathcal{S}$-VAE优于常规VAE，具有更好的重建损失。

    Barrett's Esophagus (BE) is the only precursor known to Esophageal Adenocarcinoma (EAC), a type of esophageal cancer with poor prognosis upon diagnosis. Therefore, diagnosing BE is crucial in preventing and treating esophageal cancer. While supervised machine learning supports BE diagnosis, high interobserver variability in histopathological training data limits these methods. Unsupervised representation learning via Variational Autoencoders (VAEs) shows promise, as they map input data to a lower-dimensional manifold with only useful features, characterizing BE progression for improved downstream tasks and insights. However, the VAE's Euclidean latent space distorts point relationships, hindering disease progression modeling. Geometric VAEs provide additional geometric structure to the latent space, with RHVAE assuming a Riemannian manifold and $\mathcal{S}$-VAE a hyperspherical manifold. Our study shows that $\mathcal{S}$-VAE outperforms vanilla VAE with better reconstruction losses, 
    
[^5]: 高效的多阶门控聚合网络

    Efficient Multi-order Gated Aggregation Network. (arXiv:2211.03295v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.03295](http://arxiv.org/abs/2211.03295)

    本文探索了现代卷积神经网络的表征能力，使用多阶博弈论交互的新视角，提出了一种新的纯卷积神经网络架构MogaNet，它表现出优异的可扩展性，并在多种典型视觉基准中以更高效的参数利用达到了与最先进模型竞争的效果。

    

    自从视觉变换器（ViTs）取得最近的成功之后，对ViT风格架构的探索引发了卷积神经网络的复兴。在本文中，我们从多阶博弈论交互的新视角探索了现代卷积神经网络的表征能力，这种交互反映了基于博弈论的不同尺度上下文的变量相互作用效应。在现代卷积神经网络框架内，我们使用概念上简单而有效的深度可分离卷积来定制两个特征混合器，以促进跨空间和通道空间的中阶信息。在这个基础上，提出了一种新的纯卷积神经网络架构，称为MogaNet，它表现出优异的可扩展性，并在ImageNet和包括COCO目标检测、ADE20K语义分割、2D&3D人体姿势估计以及视频预测等多种典型视觉基准中以更高效的参数利用达到了与最先进模型竞争的效果。

    Since the recent success of Vision Transformers (ViTs), explorations toward ViT-style architectures have triggered the resurgence of ConvNets. In this work, we explore the representation ability of modern ConvNets from a novel view of multi-order game-theoretic interaction, which reflects inter-variable interaction effects w.r.t.~contexts of different scales based on game theory. Within the modern ConvNet framework, we tailor the two feature mixers with conceptually simple yet effective depthwise convolutions to facilitate middle-order information across spatial and channel spaces respectively. In this light, a new family of pure ConvNet architecture, dubbed MogaNet, is proposed, which shows excellent scalability and attains competitive results among state-of-the-art models with more efficient use of parameters on ImageNet and multifarious typical vision benchmarks, including COCO object detection, ADE20K semantic segmentation, 2D\&3D human pose estimation, and video prediction. Typica
    

