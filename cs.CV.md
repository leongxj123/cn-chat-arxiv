# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Revolutionizing Disease Diagnosis with simultaneous functional PET/MR and Deeply Integrated Brain Metabolic, Hemodynamic, and Perfusion Networks](https://arxiv.org/abs/2403.20058) | 提出了MX-ARM，一种基于AI的疾病诊断模型，利用同时功能PET/MR技术，能够在推理过程中同时接受单模态和多模态输入，具有创新的模态分离和重构功能。 |
| [^2] | [Contact-aware Human Motion Generation from Textual Descriptions](https://arxiv.org/abs/2403.15709) | 本研究提出了一种新的方法CATMO，通过整合物理接触信息，从文本描述中生成视觉自然且物理合理的3D人体动作。 |
| [^3] | [CoReEcho: Continuous Representation Learning for 2D+time Echocardiography Analysis](https://arxiv.org/abs/2403.10164) | CoReEcho提出了针对直接EF回归的连续表示学习框架，在最大的超声心动图数据集上表现优越。 |
| [^4] | [Model Selection of Zero-shot Anomaly Detectors in the Absence of Labeled Validation Data](https://arxiv.org/abs/2310.10461) | 本研究提出了一个通用框架SWSA（Selection With Synthetic Anomalies），用于在没有标签验证数据的情况下选择基于图像的零样本异常检测器。通过生成合成验证集，该方法能够实现模型选择，并在实证研究中展示了比基线方法更高的AUROC。 |
| [^5] | [VideoDrafter: Content-Consistent Multi-Scene Video Generation with LLM.](http://arxiv.org/abs/2401.01256) | VideoDrafter是一个利用LLM实现内容一致的多场景视频生成的框架，能够根据输入提示生成逻辑连贯的多场景脚本，并生成高质量的视频。 |
| [^6] | [PRE: Vision-Language Prompt Learning with Reparameterization Encoder.](http://arxiv.org/abs/2309.07760) | 这项工作提出了一种名为PRE的方法，通过重新参数化编码器来增强可学习提示的泛化能力，从而解决了大型预训练视觉-语言模型中手动提示工程的挑战。 |
| [^7] | [RoCOCO: Robust Benchmark MS-COCO to Stress-test Robustness of Image-Text Matching Models.](http://arxiv.org/abs/2304.10727) | 本文提出了一个新的评估基准来测试ITM模型的鲁棒性，通过将一些“愚弄”的图片和标题添加到检索池中，在MS COCO数据集上为各种最先进的模型进行鲁棒性测试，揭示了它们的不足之处。 |
| [^8] | [GraphMLP: A Graph MLP-Like Architecture for 3D Human Pose Estimation.](http://arxiv.org/abs/2206.06420) | 提出了一种名为GraphMLP的图形增强的MLP式架构，它将图形结构纳入MLP模型中，以满足3D人体姿态的领域特定需求，同时允许局部和全局的空间交互作用。在此基础上，还将GraphMLP灵活高效地扩展到视频领域，并成功地进行了时间动力学的建模。 |

# 详细

[^1]: 利用同时功能PET/MR和深度整合的脑代谢、血液动力学和灌注网络彻底改变疾病诊断

    Revolutionizing Disease Diagnosis with simultaneous functional PET/MR and Deeply Integrated Brain Metabolic, Hemodynamic, and Perfusion Networks

    [https://arxiv.org/abs/2403.20058](https://arxiv.org/abs/2403.20058)

    提出了MX-ARM，一种基于AI的疾病诊断模型，利用同时功能PET/MR技术，能够在推理过程中同时接受单模态和多模态输入，具有创新的模态分离和重构功能。

    

    同时功能PET/MR（sf-PET/MR）是一种尖端的多模式神经影像技术。它提供了一个前所未有的机会，可以同时监测和整合由时空协变代谢活动、神经活动和脑血流（灌注）构建的多方面大脑网络。虽然在科学/临床价值上很高，但PET/MR硬件的可及性不足阻碍了其应用，更不用说现代基于AI的PET/MR融合模型。我们的目标是开发一个基于AI的临床可行疾病诊断模型，该模型基于全面的sf-PET/MR数据进行训练，在推理过程中具有允许单模态输入（例如，仅PET）以及强制多模态准确性的能力。为此，我们提出了MX-ARM，一种多模态专家混合对齐和重构模型。它是模态可分离和可交换的，动态分配不同的多层感知器（"混合）

    arXiv:2403.20058v1 Announce Type: cross  Abstract: Simultaneous functional PET/MR (sf-PET/MR) presents a cutting-edge multimodal neuroimaging technique. It provides an unprecedented opportunity for concurrently monitoring and integrating multifaceted brain networks built by spatiotemporally covaried metabolic activity, neural activity, and cerebral blood flow (perfusion). Albeit high scientific/clinical values, short in hardware accessibility of PET/MR hinders its applications, let alone modern AI-based PET/MR fusion models. Our objective is to develop a clinically feasible AI-based disease diagnosis model trained on comprehensive sf-PET/MR data with the power of, during inferencing, allowing single modality input (e.g., PET only) as well as enforcing multimodal-based accuracy. To this end, we propose MX-ARM, a multimodal MiXture-of-experts Alignment and Reconstruction Model. It is modality detachable and exchangeable, allocating different multi-layer perceptrons dynamically ("mixture 
    
[^2]: 从文本描述生成考虑接触的人体动作

    Contact-aware Human Motion Generation from Textual Descriptions

    [https://arxiv.org/abs/2403.15709](https://arxiv.org/abs/2403.15709)

    本研究提出了一种新的方法CATMO，通过整合物理接触信息，从文本描述中生成视觉自然且物理合理的3D人体动作。

    

    本文解决了从文本生成3D交互式人体动作的问题。给定描述了不同身体部位接触物体动作的文本描述，我们综合生成视觉自然且物理合理的3D人体姿势序列。然而，这个任务存在一个重要挑战，即在动作和文本描述中对物理接触的互动考虑不足，导致序列不自然且不合理。为了解决这一挑战，我们创建了一个名为RICH-CAT的新数据集，表示从RICH数据集构建的“考虑接触”的文本。RICH-CAT包括高质量动作、准确的人-物接触标签和详细的文本描述，涵盖了26种室内/室外动作的8500多对动作-文本配对。利用RICH-CAT，我们提出了一种名为CATMO的新方法，用于文本驱动的交互式人体动作合成，明确整合了物理接触的信息。

    arXiv:2403.15709v1 Announce Type: cross  Abstract: This paper addresses the problem of generating 3D interactive human motion from text. Given a textual description depicting the actions of different body parts in contact with objects, we synthesize sequences of 3D body poses that are visually natural and physically plausible. Yet, this task poses a significant challenge due to the inadequate consideration of interactions by physical contacts in both motion and textual descriptions, leading to unnatural and implausible sequences. To tackle this challenge, we create a novel dataset named RICH-CAT, representing ``Contact-Aware Texts'' constructed from the RICH dataset. RICH-CAT comprises high-quality motion, accurate human-object contact labels, and detailed textual descriptions, encompassing over 8,500 motion-text pairs across 26 indoor/outdoor actions. Leveraging RICH-CAT, we propose a novel approach named CATMO for text-driven interactive human motion synthesis that explicitly integra
    
[^3]: CoReEcho: 2D+时间超声心动图分析的连续表示学习

    CoReEcho: Continuous Representation Learning for 2D+time Echocardiography Analysis

    [https://arxiv.org/abs/2403.10164](https://arxiv.org/abs/2403.10164)

    CoReEcho提出了针对直接EF回归的连续表示学习框架，在最大的超声心动图数据集上表现优越。

    

    深度学习模型一直在不同模态的医学图像分析方面取得进展，包括超声心动图，在提供全面的端到端训练流水线的同时。然而，端到端训练流水线使得学习到的表示难以解释，并且可能无法捕获超声心动图片段之间的连续关系，导致存在虚假相关性，可能对泛化能力产生负面影响。为了缓解这一问题，我们提出了CoReEcho，这是一个强调针对直接EF回归的连续表示的新型训练框架。我们的广泛实验证明CoReEcho：1）在最大的超声心动图数据集（EchoNet-Dynamic）上表现优于当前的最先进技术（SOTA），平均绝对误差为3.90和R2 o

    arXiv:2403.10164v1 Announce Type: cross  Abstract: Deep learning (DL) models have been advancing automatic medical image analysis on various modalities, including echocardiography, by offering a comprehensive end-to-end training pipeline. This approach enables DL models to regress ejection fraction (EF) directly from 2D+time echocardiograms, resulting in superior performance. However, the end-to-end training pipeline makes the learned representations less explainable. The representations may also fail to capture the continuous relation among echocardiogram clips, indicating the existence of spurious correlations, which can negatively affect the generalization. To mitigate this issue, we propose CoReEcho, a novel training framework emphasizing continuous representations tailored for direct EF regression. Our extensive experiments demonstrate that CoReEcho: 1) outperforms the current state-of-the-art (SOTA) on the largest echocardiography dataset (EchoNet-Dynamic) with MAE of 3.90 & R2 o
    
[^4]: 无标签验证数据下零样本异常检测器的模型选择

    Model Selection of Zero-shot Anomaly Detectors in the Absence of Labeled Validation Data

    [https://arxiv.org/abs/2310.10461](https://arxiv.org/abs/2310.10461)

    本研究提出了一个通用框架SWSA（Selection With Synthetic Anomalies），用于在没有标签验证数据的情况下选择基于图像的零样本异常检测器。通过生成合成验证集，该方法能够实现模型选择，并在实证研究中展示了比基线方法更高的AUROC。

    

    异常检测需要在大型无标签数据集中检测异常样本。尽管深度学习的进步和基础模型的出现产生了强大的零样本异常检测方法，但其在实践中的应用常常受到标签数据的缺乏的限制 - 在没有标签数据的情况下，无法可靠地评估其检测性能。在这项工作中，我们提出了一种通用框架SWSA（Selection With Synthetic Anomalies）来选择基于图像的异常检测器，并使用生成的合成验证集。我们提出的异常生成方法假设只有少量的正常图像支持集，并且不需要训练或微调。生成后，我们的合成验证集被用于创建模型选择的验证框架中的检测任务。在实证研究中，我们发现SWSA常常选择与真实验证集选择相匹配的模型，结果比基线方法的AUROC更高。

    Anomaly detection requires detecting abnormal samples in large unlabeled datasets. While progress in deep learning and the advent of foundation models has produced powerful zero-shot anomaly detection methods, their deployment in practice is often hindered by the lack of labeled data -- without it, their detection performance cannot be evaluated reliably. In this work, we propose SWSA (Selection With Synthetic Anomalies): a general-purpose framework to select image-based anomaly detectors with a generated synthetic validation set. Our proposed anomaly generation method assumes access to only a small support set of normal images and requires no training or fine-tuning. Once generated, our synthetic validation set is used to create detection tasks that compose a validation framework for model selection. In an empirical study, we find that SWSA often selects models that match selections made with a ground-truth validation set, resulting in higher AUROCs than baseline methods. We also find
    
[^5]: VideoDrafter: 利用LLM实现内容一致的多场景视频生成

    VideoDrafter: Content-Consistent Multi-Scene Video Generation with LLM. (arXiv:2401.01256v1 [cs.CV])

    [http://arxiv.org/abs/2401.01256](http://arxiv.org/abs/2401.01256)

    VideoDrafter是一个利用LLM实现内容一致的多场景视频生成的框架，能够根据输入提示生成逻辑连贯的多场景脚本，并生成高质量的视频。

    

    最近扩展模型的创新和突破显著扩大了根据给定提示生成高质量视频的可能性。现有的大多数作品仅处理在单个背景中发生单个视频事件的单场景情况。然而，扩展到生成多场景视频并且在保持各个场景之间的逻辑一致同时保持视觉外观一致性方面并不简单。在本文中，我们提出了一种新颖的框架，即VideoDrafter，用于内容一致的多场景视频生成。技术上，VideoDrafter利用大型语言模型（LLM）将输入提示转化为综合的多场景脚本，该脚本从LLM学到的逻辑知识中受益。每个场景的脚本包括描述事件、前景/背景实体以及摄像机运动的提示。VideoDrafter识别脚本中的共同实体，并询问LLM来选择生成逻辑连贯的视频场景。

    The recent innovations and breakthroughs in diffusion models have significantly expanded the possibilities of generating high-quality videos for the given prompts. Most existing works tackle the single-scene scenario with only one video event occurring in a single background. Extending to generate multi-scene videos nevertheless is not trivial and necessitates to nicely manage the logic in between while preserving the consistent visual appearance of key content across video scenes. In this paper, we propose a novel framework, namely VideoDrafter, for content-consistent multi-scene video generation. Technically, VideoDrafter leverages Large Language Models (LLM) to convert the input prompt into comprehensive multi-scene script that benefits from the logical knowledge learnt by LLM. The script for each scene includes a prompt describing the event, the foreground/background entities, as well as camera movement. VideoDrafter identifies the common entities throughout the script and asks LLM
    
[^6]: PRE: 视觉-语言提示学习与重新参数化编码器

    PRE: Vision-Language Prompt Learning with Reparameterization Encoder. (arXiv:2309.07760v1 [cs.CV])

    [http://arxiv.org/abs/2309.07760](http://arxiv.org/abs/2309.07760)

    这项工作提出了一种名为PRE的方法，通过重新参数化编码器来增强可学习提示的泛化能力，从而解决了大型预训练视觉-语言模型中手动提示工程的挑战。

    

    大型预训练的视觉-语言模型（如CLIP）已经展示出在零样本迁移任务中具有巨大潜力。然而，为了达到最佳性能，需要手动选择提示以改进下游图像分布和文本类描述之间的对齐。这种手动提示工程是将这些模型部署到实践中的主要挑战，因为它需要领域专业知识并且非常耗时。为了避免复杂的提示工程，最近的CoOp工作引入了在视觉领域使用可控文本标记的提示学习概念。虽然CoOp可以在手动提示上取得显著改进，但其学到的上下文在同一数据集中更广泛的未见类别中的泛化能力较差。在这项工作中，我们提出了一种名为Prompt Learning with Reparameterization Encoder (PRE) 的简单高效的方法，改进了可学习提示的泛化能力。

    Large pre-trained vision-language models such as CLIP have demonstrated great potential in zero-shot transferability to downstream tasks. However, to attain optimal performance, the manual selection of prompts is necessary to improve alignment between the downstream image distribution and the textual class descriptions. This manual prompt engineering is the major challenge for deploying such models in practice since it requires domain expertise and is extremely time-consuming. To avoid non-trivial prompt engineering, recent work Context Optimization (CoOp) introduced the concept of prompt learning to the vision domain using learnable textual tokens. While CoOp can achieve substantial improvements over manual prompts, its learned context is worse generalizable to wider unseen classes within the same dataset. In this work, we present Prompt Learning with Reparameterization Encoder (PRE) - a simple and efficient method that enhances the generalization ability of the learnable prompt to un
    
[^7]: RoCOCO：稳健的基准MS-COCO评估图文匹配模型的鲁棒性

    RoCOCO: Robust Benchmark MS-COCO to Stress-test Robustness of Image-Text Matching Models. (arXiv:2304.10727v1 [cs.CV])

    [http://arxiv.org/abs/2304.10727](http://arxiv.org/abs/2304.10727)

    本文提出了一个新的评估基准来测试ITM模型的鲁棒性，通过将一些“愚弄”的图片和标题添加到检索池中，在MS COCO数据集上为各种最先进的模型进行鲁棒性测试，揭示了它们的不足之处。

    

    近年来，大规模的视觉语言预训练模型和视觉语义嵌入方法显著提高了MS COCO 5K测试集上图文匹配（ITM）的准确性。然而，当将这些最先进的模型用于实际应用时，它们的鲁棒性仍不清楚。本文提出了一个新的评估基准来测试ITM模型的鲁棒性。为此，我们将各种“愚弄”的图片和标题添加到检索池中。具体而言，我们通过插入不相关的图像来更改图像，并通过替换名词来更改标题，从而改变句子的含义。我们发现，仅仅将这些新创建的图像和标题添加到测试集中就可以降低各种最先进模型的性能（例如，在BLIP中从81.9％降至64.5％，在VSE∞中从66.1％降至37.5％）。我们希望我们的发现能为提高视觉语言模型的鲁棒性和设计更多样化的压力测试提供启示。

    Recently, large-scale vision-language pre-training models and visual semantic embedding methods have significantly improved image-text matching (ITM) accuracy on MS COCO 5K test set. However, it is unclear how robust these state-of-the-art (SOTA) models are when using them in the wild. In this paper, we propose a novel evaluation benchmark to stress-test the robustness of ITM models. To this end, we add various fooling images and captions to a retrieval pool. Specifically, we change images by inserting unrelated images, and change captions by substituting a noun, which can change the meaning of a sentence. We discover that just adding these newly created images and captions to the test set can degrade performances (i.e., Recall@1) of a wide range of SOTA models (e.g., 81.9% $\rightarrow$ 64.5% in BLIP, 66.1% $\rightarrow$ 37.5% in VSE$\infty$). We expect that our findings can provide insights for improving the robustness of the vision-language models and devising more diverse stress-te
    
[^8]: GraphMLP：一种用于3D人体姿态估计的图形MLP式架构

    GraphMLP: A Graph MLP-Like Architecture for 3D Human Pose Estimation. (arXiv:2206.06420v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2206.06420](http://arxiv.org/abs/2206.06420)

    提出了一种名为GraphMLP的图形增强的MLP式架构，它将图形结构纳入MLP模型中，以满足3D人体姿态的领域特定需求，同时允许局部和全局的空间交互作用。在此基础上，还将GraphMLP灵活高效地扩展到视频领域，并成功地进行了时间动力学的建模。

    

    现代多层感知器（MLP）模型已经展现出在没有自我注意力的情况下学习视觉表示方面的竞争性结果，然而，现有的MLP模型并不擅长捕捉局部细节，也缺乏有关人体构型的先验知识，这限制了它们用于骨骼表示学习的建模能力。为了解决这些问题，我们提出了一种简单而有效的图形增强的MLP式架构，称为GraphMLP，它结合了MLP和图形卷积网络（GCN）在全局-局部-图形统一架构中用于3D人体姿态估计。GraphMLP将人体的图形结构纳入MLP模型中，以满足3D人体姿态的领域特定需求，同时允许局部和全局的空间交互作用。此外，我们提出了将GraphMLP灵活高效地扩展到视频领域，并展示了可以以可忽略的计算代价来有效地建模复杂的时间动力学。

    Modern multi-layer perceptron (MLP) models have shown competitive results in learning visual representations without self-attention. However, existing MLP models are not good at capturing local details and lack prior knowledge of human body configurations, which limits their modeling power for skeletal representation learning. To address these issues, we propose a simple yet effective graph-reinforced MLP-Like architecture, named GraphMLP, that combines MLPs and graph convolutional networks (GCNs) in a global-local-graphical unified architecture for 3D human pose estimation. GraphMLP incorporates the graph structure of human bodies into an MLP model to meet the domain-specific demand of the 3D human pose, while allowing for both local and global spatial interactions. Furthermore, we propose to flexibly and efficiently extend the GraphMLP to the video domain and show that complex temporal dynamics can be effectively modeled in a simple way with negligible computational cost gains in the
    

