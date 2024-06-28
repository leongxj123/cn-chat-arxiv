# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Physics-Guided Neural Networks for Intraventricular Vector Flow Mapping](https://arxiv.org/abs/2403.13040) | 该研究提出了使用物理引导的神经网络和基于nnU-Net的监督方法来优化心室向量流动映射，效果与传统算法相当，且具有更高的效率和泛化能力。 |
| [^2] | [Training Small Multimodal Models to Bridge Biomedical Competency Gap: A Case Study in Radiology Imaging](https://arxiv.org/abs/2403.08002) | 本文针对生物医学应用中前沿模型尚存在的多模态能力差距，探讨了训练开源小型多模态模型以弥补临床需求的生物医学能力差距。 |
| [^3] | [Shortcut Learning in Medical Image Segmentation](https://arxiv.org/abs/2403.06748) | 本研究将捷径学习现象扩展到医学图像分割领域，发现临床注释和特定数据处理方式可能误导模型并影响分割准确性，提出了缓解捷径学习影响的策略。 |
| [^4] | [Taking Training Seriously: Human Guidance and Management-Based Regulation of Artificial Intelligence](https://arxiv.org/abs/2402.08466) | 这项研究认为，在人工智能的管理式监管方法中，加强人类引导和培训技术的研究和实践对于提高AI的性能，解决技术和伦理问题等方面具有重要作用。 |
| [^5] | [Source-Free Domain Adaptation with Diffusion-Guided Source Data Generation](https://arxiv.org/abs/2402.04929) | 本文提出了一种无源域自适应的新方法，利用扩散模型生成上下文相关的领域特定图像，通过微调预训练模型和无监督领域自适应技术实现了显著的性能改进。 |
| [^6] | [AdaTreeFormer: Few Shot Domain Adaptation for Tree Counting from a Single High-Resolution Image](https://arxiv.org/abs/2402.02956) | AdaTreeFormer是一种从源领域学习并适应只有有限数量标注树木的目标领域的框架，利用一个共享的编码器和分层特征提取方案，实现了树木计数的少样本领域自适应。 |
| [^7] | [Transfer Learning in ECG Diagnosis: Is It Effective?](https://arxiv.org/abs/2402.02021) | 本研究首次对心电图诊断中的迁移学习进行了广泛的经验研究，发现微调对于小型数据集是较好的选择，当数据集足够大时，从头开始训练可以达到可比性能，但需要更长的训练时间。同时，迁移学习与卷积神经网络具有更好的兼容性。 |
| [^8] | [Muffin or Chihuahua? Challenging Large Vision-Language Models with Multipanel VQA](https://arxiv.org/abs/2401.15847) | 引入了多面板视觉问答（MultipanelVQA）基准挑战大型视觉语言模型（LVLMs）对理解多面板图像的能力，并发现LVLMs在这方面仍然存在显著挑战。 |
| [^9] | [WsiCaption: Multiple Instance Generation of Pathology Reports for Gigapixel Whole-Slide Images](https://arxiv.org/abs/2311.16480) | 研究提出了一种基于多实例生成模型的方法，能够生成千亿像素全切片图像的病理报告，实验结果表明该模型能够产生包含多个临床线索的病理报告。 |
| [^10] | [Challenging Common Paradigms in Multi-Task Learning](https://arxiv.org/abs/2311.04698) | 我们挑战了多任务学习中的常见范式，通过研究在单任务学习中的影响，揭示了优化器选择在MTL中的关键作用，并理论推导出了梯度冲突的角色。 |
| [^11] | [MixerFlow for Image Modelling.](http://arxiv.org/abs/2310.16777) | MixerFlow是一种新型的基于MLP-Mixer架构的正则化流模型，通过提供有效的权重共享机制，实现了更好的图像密度估计性能和更丰富的嵌入表示。 |
| [^12] | [Auto-Prompting SAM for Mobile Friendly 3D Medical Image Segmentation.](http://arxiv.org/abs/2308.14936) | 这项工作提出了一种名为AutoSAM Adapter的方法，用于解决SAM在3D医学图像分割任务上的性能问题。通过参数高效的适应技术，实现了自动提示学习范式，消除了对手动生成提示的需求。 |

# 详细

[^1]: 物理引导的神经网络用于心室向量流动映射

    Physics-Guided Neural Networks for Intraventricular Vector Flow Mapping

    [https://arxiv.org/abs/2403.13040](https://arxiv.org/abs/2403.13040)

    该研究提出了使用物理引导的神经网络和基于nnU-Net的监督方法来优化心室向量流动映射，效果与传统算法相当，且具有更高的效率和泛化能力。

    

    Intraventricular vector flow mapping (iVFM)旨在增强和量化心脏成像中的彩色多普勒。本研究提出了一种新颖的替代方案，通过利用物理信息神经网络（PINNs）和基于物理引导的nnU-Net监督方法来优化传统的iVFM优化方案。通过对基于患者特定计算流体动力学模型产生的模拟彩色多普勒图像和体内多普勒采集的严格评估，两种方法均展现出与原始iVFM算法相当的重建性能。 PINNs的效率通过双阶段优化和预优化权重得到提升。另一方面，nnU-Net方法在泛化能力和实时性能方面表现出色。值得注意的是，nnU-Net在稀疏和截断多普勒数据上表现出更好的鲁棒性，同时保持独立于明确的边界条件。总的来说，我们的结果突出了效果

    arXiv:2403.13040v1 Announce Type: cross  Abstract: Intraventricular vector flow mapping (iVFM) seeks to enhance and quantify color Doppler in cardiac imaging. In this study, we propose novel alternatives to the traditional iVFM optimization scheme by utilizing physics-informed neural networks (PINNs) and a physics-guided nnU-Net-based supervised approach. Through rigorous evaluation on simulated color Doppler images derived from a patient-specific computational fluid dynamics model and in vivo Doppler acquisitions, both approaches demonstrate comparable reconstruction performance to the original iVFM algorithm. The efficiency of PINNs is boosted through dual-stage optimization and pre-optimized weights. On the other hand, the nnU-Net method excels in generalizability and real time capabilities. Notably, nnU-Net shows superior robustness on sparse and truncated Doppler data while maintaining independence from explicit boundary conditions. Overall, our results highlight the effectiveness
    
[^2]: 训练小型多模态模型以填补生物医学能力差距：以放射学成像为例

    Training Small Multimodal Models to Bridge Biomedical Competency Gap: A Case Study in Radiology Imaging

    [https://arxiv.org/abs/2403.08002](https://arxiv.org/abs/2403.08002)

    本文针对生物医学应用中前沿模型尚存在的多模态能力差距，探讨了训练开源小型多模态模型以弥补临床需求的生物医学能力差距。

    

    放大基础模型的尺度规律和非凡表现激励了在生物医学领域开发和利用这些大型模型。然而，尽管在一些生物医学基准测试中取得了早期有希望的结果，但在这些模型能够应用于真实世界的应用之前仍然存在一些重大挑战。像GPT-4V这样的前沿模型在生物医学应用中仍存在重大的多模态能力差距。此外，访问、成本、延迟和合规等实际问题使临床医生难以直接在私人患者数据上使用私人托管的最先进大型模型。在本文中，我们探讨训练开源小型多模态模型（SMMs）来填补未满足的临床需求的生物医学能力差距。为了最大化数据效率，我们采用模块化方法，将用于图像和文本模态的最先进预训练模型纳入，并侧重于t

    arXiv:2403.08002v1 Announce Type: new  Abstract: The scaling laws and extraordinary performance of large foundation models motivate the development and utilization of such large models in biomedicine. However, despite early promising results on some biomedical benchmarks, there are still major challenges that need to be addressed before these models can be used in real-world applications. Frontier models such as GPT-4V still have major competency gaps in multimodal capabilities for biomedical applications. Moreover, pragmatic issues such as access, cost, latency, and compliance make it hard for clinicians to use privately-hosted state-of-the-art large models directly on private patient data. In this paper, we explore training open-source small multimodal models (SMMs) to bridge biomedical competency gaps for unmet clinical needs. To maximize data efficiency, we adopt a modular approach by incorporating state-of-the-art pre-trained models for image and text modalities, and focusing on t
    
[^3]: 医学图像分割中的捷径学习

    Shortcut Learning in Medical Image Segmentation

    [https://arxiv.org/abs/2403.06748](https://arxiv.org/abs/2403.06748)

    本研究将捷径学习现象扩展到医学图像分割领域，发现临床注释和特定数据处理方式可能误导模型并影响分割准确性，提出了缓解捷径学习影响的策略。

    

    捷径学习是一种现象，机器学习模型优先学习简单、潜在误导的数据提示，这些提示在训练集之外很难泛化。现有研究主要探讨这一现象在图像分类领域，而这项研究将捷径学习探索延伸到医学图像分割中。我们证明，临床注释如卡尺，以及数据集中零填充卷积和中心裁剪的组合可以无意中作为捷径，影响分割准确性。我们在两个不同但常见的医学图像分割任务中识别和评估了捷径学习。此外，我们提出了缓解捷径学习影响、提高分割模型泛化能力的策略。通过揭示医学图像分割中捷径的存在和影响，我们提供了一些见解。

    arXiv:2403.06748v1 Announce Type: cross  Abstract: Shortcut learning is a phenomenon where machine learning models prioritize learning simple, potentially misleading cues from data that do not generalize well beyond the training set. While existing research primarily investigates this in the realm of image classification, this study extends the exploration of shortcut learning into medical image segmentation. We demonstrate that clinical annotations such as calipers, and the combination of zero-padded convolutions and center-cropped training sets in the dataset can inadvertently serve as shortcuts, impacting segmentation accuracy. We identify and evaluate the shortcut learning on two different but common medical image segmentation tasks. In addition, we suggest strategies to mitigate the influence of shortcut learning and improve the generalizability of the segmentation models. By uncovering the presence and implications of shortcuts in medical image segmentation, we provide insights a
    
[^4]: 认真对待培训：人工智能的人类引导与基于管理的监管

    Taking Training Seriously: Human Guidance and Management-Based Regulation of Artificial Intelligence

    [https://arxiv.org/abs/2402.08466](https://arxiv.org/abs/2402.08466)

    这项研究认为，在人工智能的管理式监管方法中，加强人类引导和培训技术的研究和实践对于提高AI的性能，解决技术和伦理问题等方面具有重要作用。

    

    对人工智能（AI）相关危害更强大的治理的热情呼声正在世界范围内引起管理学者所称的基于管理的监管方法的采用。美国和欧洲的最新倡议以及国际标准化组织采纳的重要自我监管标准都共同具有一个核心的基于管理的范式。这些基于管理的倡议旨在通过增加人类对AI工具的培训和开发的监督来激励。因此，在这个新兴的基于管理的监管范式时代中，需要对人类引导培训技术进行完善和系统化。如果认真对待，人类引导培训可以减轻一些对AI的技术和伦理压力，以人类直觉提高AI的性能，并更好地满足对公平性和有效解释的需求。在本文中，我们讨论了连接。

    Fervent calls for more robust governance of the harms associated with artificial intelligence (AI) are leading to the adoption around the world of what regulatory scholars have called a management-based approach to regulation. Recent initiatives in the United States and Europe, as well as the adoption of major self-regulatory standards by the International Organization for Standardization, share in common a core management-based paradigm. These management-based initiatives seek to motivate an increase in human oversight of how AI tools are trained and developed. Refinements and systematization of human-guided training techniques will thus be needed to fit within this emerging era of management-based regulatory paradigm. If taken seriously, human-guided training can alleviate some of the technical and ethical pressures on AI, boosting AI performance with human intuition as well as better addressing the needs for fairness and effective explainability. In this paper, we discuss the connec
    
[^5]: 无源域自适应的扩散引导源数据生成

    Source-Free Domain Adaptation with Diffusion-Guided Source Data Generation

    [https://arxiv.org/abs/2402.04929](https://arxiv.org/abs/2402.04929)

    本文提出了一种无源域自适应的新方法，利用扩散模型生成上下文相关的领域特定图像，通过微调预训练模型和无监督领域自适应技术实现了显著的性能改进。

    

    本文引入了一种利用扩散模型的泛化能力进行无源域自适应（DM-SFDA）的新方法。我们提出的DM-SFDA方法包括对预训练的文本到图像扩散模型进行微调，并使用目标图像的特征来指导扩散过程生成源域图像。具体而言，预训练的扩散模型被微调以生成最小化熵并最大化预训练源模型置信度的源样本。然后，我们应用已建立的无监督领域自适应技术将生成的源图像与目标域数据进行对齐。我们通过在一系列数据集上进行全面实验验证了我们的方法，包括Office-31、Office-Home和VisDA。结果显示，在无源域自适应的性能方面取得了显著的改进，展示了扩散模型在生成上下文相关的、领域特定的图像方面的潜力。

    This paper introduces a novel approach to leverage the generalizability capability of Diffusion Models for Source-Free Domain Adaptation (DM-SFDA). Our proposed DM-SFDA method involves fine-tuning a pre-trained text-to-image diffusion model to generate source domain images using features from the target images to guide the diffusion process. Specifically, the pre-trained diffusion model is fine-tuned to generate source samples that minimize entropy and maximize confidence for the pre-trained source model. We then apply established unsupervised domain adaptation techniques to align the generated source images with target domain data. We validate our approach through comprehensive experiments across a range of datasets, including Office-31, Office-Home, and VisDA. The results highlight significant improvements in SFDA performance, showcasing the potential of diffusion models in generating contextually relevant, domain-specific images.
    
[^6]: AdaTreeFormer: 从一张高分辨率图像中进行树木计数的少样本领域自适应

    AdaTreeFormer: Few Shot Domain Adaptation for Tree Counting from a Single High-Resolution Image

    [https://arxiv.org/abs/2402.02956](https://arxiv.org/abs/2402.02956)

    AdaTreeFormer是一种从源领域学习并适应只有有限数量标注树木的目标领域的框架，利用一个共享的编码器和分层特征提取方案，实现了树木计数的少样本领域自适应。

    

    仅使用一张航空或卫星图像来估计和计数树木密度是摄影测量和遥感领域中一项困难的任务。然而，它在森林管理中起着至关重要的作用。不同地形上各种各样的树木种类严重阻碍了树木计数模型的良好表现。本文旨在提出一个从具有足够标注树木的源领域学习并适应只有有限数量标注树木的目标领域的框架。我们的方法称为AdaTreeFormer，包含一个共享的编码器和一个分层特征提取方案，用于从源领域和目标领域中提取稳健的特征。它还包括三个子网络：两个用于分别从源领域和目标领域提取自注意力图，并一个用于提取跨领域注意力图。对于后者，引入了一种注意力适应机制，用于从不同领域中提取相关信息。

    The process of estimating and counting tree density using only a single aerial or satellite image is a difficult task in the fields of photogrammetry and remote sensing. However, it plays a crucial role in the management of forests. The huge variety of trees in varied topography severely hinders tree counting models to perform well. The purpose of this paper is to propose a framework that is learnt from the source domain with sufficient labeled trees and is adapted to the target domain with only a limited number of labeled trees. Our method, termed as AdaTreeFormer, contains one shared encoder with a hierarchical feature extraction scheme to extract robust features from the source and target domains. It also consists of three subnets: two for extracting self-domain attention maps from source and target domains respectively and one for extracting cross-domain attention maps. For the latter, an attention-to-adapt mechanism is introduced to distill relevant information from different doma
    
[^7]: ECG诊断中的迁移学习：有效吗？

    Transfer Learning in ECG Diagnosis: Is It Effective?

    [https://arxiv.org/abs/2402.02021](https://arxiv.org/abs/2402.02021)

    本研究首次对心电图诊断中的迁移学习进行了广泛的经验研究，发现微调对于小型数据集是较好的选择，当数据集足够大时，从头开始训练可以达到可比性能，但需要更长的训练时间。同时，迁移学习与卷积神经网络具有更好的兼容性。

    

    在真实世界的场景中，深度学习在心电图诊断中的应用往往受到大规模、标记良好的数据集的稀缺性的限制，因此使用迁移学习来利用从更大的数据集中学到的特征。然而，关于迁移学习始终优于从头开始训练的普遍假设从未被系统验证过。在本研究中，我们通过对多标签心电图分类中进行微调与从头开始训练的性能进行比较，涵盖了各种心电图数据集和深度神经网络，进行了第一次广泛的经验性研究来验证迁移学习的有效性。我们证实，对于小型的下游数据集来说，微调是更好的选择；然而，当数据集足够大时，从头开始训练可以达到可比性能，尽管需要更长的训练时间来迎头赶上。此外，我们发现，迁移学习与卷积神经网络更好的兼容性。

    The adoption of deep learning in ECG diagnosis is often hindered by the scarcity of large, well-labeled datasets in real-world scenarios, leading to the use of transfer learning to leverage features learned from larger datasets. Yet the prevailing assumption that transfer learning consistently outperforms training from scratch has never been systematically validated. In this study, we conduct the first extensive empirical study on the effectiveness of transfer learning in multi-label ECG classification, by investigating comparing the fine-tuning performance with that of training from scratch, covering a variety of ECG datasets and deep neural networks. We confirm that fine-tuning is the preferable choice for small downstream datasets; however, when the dataset is sufficiently large, training from scratch can achieve comparable performance, albeit requiring a longer training time to catch up. Furthermore, we find that transfer learning exhibits better compatibility with convolutional ne
    
[^8]: 松饼还是吉娃娃？用多面板VQA挑战大型视觉语言模型

    Muffin or Chihuahua? Challenging Large Vision-Language Models with Multipanel VQA

    [https://arxiv.org/abs/2401.15847](https://arxiv.org/abs/2401.15847)

    引入了多面板视觉问答（MultipanelVQA）基准挑战大型视觉语言模型（LVLMs）对理解多面板图像的能力，并发现LVLMs在这方面仍然存在显著挑战。

    

    多面板图像，通常在网页截图、海报等中看到，充斥着我们的日常生活。这些图像以多个子图以不同布局组成，有效地向人们传达信息。为了构建高级的多模态人工智能应用，如能理解复杂场景并在网页中导航的代理程序，多面板视觉推理的技能是至关重要的，对模型在这方面进行全面评估是很重要的。因此，我们引入了多面板视觉问答（MultipanelVQA），这是一个新颖的基准，包括6,600个问题、答案和多面板图像三元组，专门挑战模型理解多面板图像。我们的评估表明，MultipanelVQA基准中的问题对测试的最先进的大型视觉语言模型（LVLMs）提出了重大挑战，即使人类可以获得约99%的准确率。

    arXiv:2401.15847v2 Announce Type: replace-cross  Abstract: Multipanel images, commonly seen as web screenshots, posters, etc., pervade our daily lives. These images, characterized by their composition of multiple subfigures in distinct layouts, effectively convey information to people. Toward building advanced multimodal AI applications, such as agents that understand complex scenes and navigate through webpages, the skill of multipanel visual reasoning is essential, and a comprehensive evaluation of models in this regard is important. Therefore, we introduce Multipanel Visual Question Answering (MultipanelVQA), a novel benchmark comprising 6,600 triplets of questions, answers, and multipanel images that specifically challenge models in comprehending multipanel images. Our evaluation shows that questions in the MultipanelVQA benchmark pose significant challenges to the state-of-the-art Large Vision Language Models (LVLMs) tested, even though humans can attain approximately 99\% accurac
    
[^9]: 病理报告的多实例生成用于千亿像素全切片图像

    WsiCaption: Multiple Instance Generation of Pathology Reports for Gigapixel Whole-Slide Images

    [https://arxiv.org/abs/2311.16480](https://arxiv.org/abs/2311.16480)

    研究提出了一种基于多实例生成模型的方法，能够生成千亿像素全切片图像的病理报告，实验结果表明该模型能够产生包含多个临床线索的病理报告。

    

    全切片图像是用于癌症诊断和治疗的数字病理学的基础。撰写病理报告对经验不足的病理学家来说是费时且容易出错的。为了减少工作量并改善临床自动化，我们研究了如何生成给定全切片图像的病理报告。在数据端，我们整理了最大的WSI-文本数据集（TCGA-PathoText）。具体来说，我们通过识别和清理TCGA中叙述诊断幻灯片的病理报告，收集了近1万对高质量的WSI-文本配对，供视觉-语言模型使用。在模型端，我们提出了可以为千亿像素WSI生成病理报告的多实例生成模型（MI-Gen）。我们在TCGA-PathoText的最大子集上对我们的模型进行了基准测试。实验结果表明，我们的模型可以生成包含多个临床线索的病理报告。此外，WSI-文本预测可被视为一种方法。

    arXiv:2311.16480v2 Announce Type: replace-cross  Abstract: Whole slide images are the foundation of digital pathology for the diagnosis and treatment of carcinomas. Writing pathology reports is laborious and error-prone for inexperienced pathologists. To reduce the workload and improve clinical automation, we investigate how to generate pathology reports given whole slide images. On the data end, we curated the largest WSI-text dataset (TCGA-PathoText). In specific, we collected nearly 10000 high-quality WSI-text pairs for visual-language models by recognizing and cleaning pathology reports which narrate diagnostic slides in TCGA. On the model end, we propose the multiple instance generative model (MI-Gen) which can produce pathology reports for gigapixel WSIs. We benchmark our model on the largest subset of TCGA-PathoText. Experimental results show our model can generate pathology reports which contain multiple clinical clues. Furthermore, WSI-text prediction can be seen as an approac
    
[^10]: 在多任务学习中挑战常见范式

    Challenging Common Paradigms in Multi-Task Learning

    [https://arxiv.org/abs/2311.04698](https://arxiv.org/abs/2311.04698)

    我们挑战了多任务学习中的常见范式，通过研究在单任务学习中的影响，揭示了优化器选择在MTL中的关键作用，并理论推导出了梯度冲突的角色。

    

    尽管近年来多任务学习（MTL）受到了极大关注，但其基本机制仍然知之甚少。最近的方法并未带来一致的性能改进，相比单任务学习（STL）基线，强调了更深入了解MTL特定挑战的重要性。在我们的研究中，我们挑战了MTL中的范式，提出了几点关于STL的重要影响：首先，优化器的选择对MTL的影响只受到了轻微的调查。我们通过各种实验的实证方法展示了常见STL工具（例如Adam优化器）在MTL中的关键作用。为了进一步研究Adam的有效性，我们在一定的假设下从理论上推导出部分损失尺度不变性。其次，梯度冲突的概念经常被描述为MTL中的一个特定问题。我们深入探讨了梯度冲突在MTL中的作用，并将其与STL进行比较。在角度梯度对齐方面，我们没有找到

    arXiv:2311.04698v3 Announce Type: replace-cross  Abstract: While multi-task learning (MTL) has gained significant attention in recent years, its underlying mechanisms remain poorly understood. Recent methods did not yield consistent performance improvements over single task learning (STL) baselines, underscoring the importance of gaining more profound insights about challenges specific to MTL. In our study, we challenge paradigms in MTL in the context of STL: First, the impact of the choice of optimizer has only been mildly investigated in MTL. We show the pivotal role of common STL tools such as the Adam optimizer in MTL empirically in various experiments. To further investigate Adam's effectiveness, we theoretical derive a partial loss-scale invariance under mild assumptions. Second, the notion of gradient conflicts has often been phrased as a specific problem in MTL. We delve into the role of gradient conflicts in MTL and compare it to STL. For angular gradient alignment we find no 
    
[^11]: 图像建模的MixerFlow

    MixerFlow for Image Modelling. (arXiv:2310.16777v1 [stat.ML])

    [http://arxiv.org/abs/2310.16777](http://arxiv.org/abs/2310.16777)

    MixerFlow是一种新型的基于MLP-Mixer架构的正则化流模型，通过提供有效的权重共享机制，实现了更好的图像密度估计性能和更丰富的嵌入表示。

    

    正则化流是一种统计模型，通过使用双射变换将复杂密度转换为简单密度，实现了密度估计和从单个模型生成数据的功能。在图像建模的背景下，主要选择的是基于Glow的架构，而其他架构在研究界尚未得到广泛探索。在本研究中，我们提出了一种基于MLP-Mixer架构的新型架构MixerFlow，进一步统一了生成性和判别性建模架构。MixerFlow提供了一种有效的权重共享机制，适用于基于流的模型。我们的结果表明，在固定计算预算下，MixerFlow在图像数据集上具有更好的密度估计性能，并且随着图像分辨率的增加，其性能也得到了良好的扩展，使得MixerFlow成为Glow-based架构的一个强大而简单的替代品。我们还展示了MixerFlow提供了比Glow-based架构更丰富的嵌入表示。

    Normalising flows are statistical models that transform a complex density into a simpler density through the use of bijective transformations enabling both density estimation and data generation from a single model. In the context of image modelling, the predominant choice has been the Glow-based architecture, whereas alternative architectures remain largely unexplored in the research community. In this work, we propose a novel architecture called MixerFlow, based on the MLP-Mixer architecture, further unifying the generative and discriminative modelling architectures. MixerFlow offers an effective mechanism for weight sharing for flow-based models. Our results demonstrate better density estimation on image datasets under a fixed computational budget and scales well as the image resolution increases, making MixeFlow a powerful yet simple alternative to the Glow-based architectures. We also show that MixerFlow provides more informative embeddings than Glow-based architectures.
    
[^12]: 为移动友好的3D医学图像分割自动提示SAM

    Auto-Prompting SAM for Mobile Friendly 3D Medical Image Segmentation. (arXiv:2308.14936v1 [cs.CV])

    [http://arxiv.org/abs/2308.14936](http://arxiv.org/abs/2308.14936)

    这项工作提出了一种名为AutoSAM Adapter的方法，用于解决SAM在3D医学图像分割任务上的性能问题。通过参数高效的适应技术，实现了自动提示学习范式，消除了对手动生成提示的需求。

    

    Segment Anything Model (SAM)已经被迅速应用于各种自然图像的分割。然而，最近的研究表明，SAM在3D医学图像分割任务上的性能不佳。除了自然图像和医学图像之间的领域差距外，2D和3D图像之间的空间布局差异，强大的GPU服务器所带来的大量计算负担，以及耗时的手动提示生成使得SAM无法扩展到更广泛的医学图像分割应用。为了解决这些挑战，在这项工作中，我们引入了一种新方法AutoSAM Adapter，专为3D多器官CT分割而设计。我们采用参数高效的适应技术开发了一种自动提示学习范式，以促进将SAM模型的能力转化为3D医学图像分割，消除了手动生成提示的需求。

    The Segment Anything Model (SAM) has rapidly been adopted for segmenting a wide range of natural images. However, recent studies have indicated that SAM exhibits subpar performance on 3D medical image segmentation tasks. In addition to the domain gaps between natural and medical images, disparities in the spatial arrangement between 2D and 3D images, the substantial computational burden imposed by powerful GPU servers, and the time-consuming manual prompt generation impede the extension of SAM to a broader spectrum of medical image segmentation applications. To address these challenges, in this work, we introduce a novel method, AutoSAM Adapter, designed specifically for 3D multi-organ CT-based segmentation. We employ parameter-efficient adaptation techniques in developing an automatic prompt learning paradigm to facilitate the transformation of the SAM model's capabilities to 3D medical image segmentation, eliminating the need for manually generated prompts. Furthermore, we effectivel
    

