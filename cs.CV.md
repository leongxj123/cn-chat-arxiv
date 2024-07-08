# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multimodal Variational Autoencoder for Low-cost Cardiac Hemodynamics Instability Detection](https://arxiv.org/abs/2403.13658) | 提出了一种新颖的多模态变分自编码器（$\text{CardioVAE}_\text{X,G}$），将低成本胸部X射线（CXR）和心电图（ECG）数据形式整合起来，并实现了共享特征和独特特征的学习。 |
| [^2] | [Hierarchical Gaussian Mixture Normalizing Flow Modeling for Unified Anomaly Detection](https://arxiv.org/abs/2403.13349) | 提出了一种用于统一异常检测的Hierarchical Gaussian mixture normalizing flow (HGAD)建模方法，通过分层高斯混合建模来提升异常检测模型的表示能力 |
| [^3] | [DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation](https://arxiv.org/abs/2403.07788) | DexCap是一个可移植的手部动作捕捉系统，结合DexIL算法从人类手部运动数据中训练机器人技能，具有精确追踪和复制人类动作的能力。 |
| [^4] | [Towards Multimodal Sentiment Analysis Debiasing via Bias Purification](https://arxiv.org/abs/2403.05023) | 提出了一种基于因果关系的多模态对事实推理情感分析框架，用于净化和缓解数据集的偏见，从而提高多模态情感分析的性能。 |
| [^5] | [U$^2$MRPD: Unsupervised undersampled MRI reconstruction by prompting a large latent diffusion model](https://arxiv.org/abs/2402.10609) | U$^2$MRPD是一个新颖的框架，通过大型潜在扩散模型引导，实现了无监督的欠采样MRI重建，能够支持图像特定的MRI重建，且在多个数据集上表现出与监督和MRI扩散方法相媲美甚至更好的性能。 |
| [^6] | [ScreenAI: A Vision-Language Model for UI and Infographics Understanding](https://arxiv.org/abs/2402.04615) | ScreenAI是一个专注于UI和信息图表理解的视觉-语言模型，通过灵活的修补策略和独特的数据集训练，以及针对UI元素的屏幕注解任务的处理，实现了在多个任务上的新的最优结果。 |
| [^7] | [MT-HCCAR: Multi-Task Deep Learning with Hierarchical Classification and Attention-based Regression for Cloud Property Retrieval.](http://arxiv.org/abs/2401.16520) | 这篇论文提出了一种名为MT-HCCAR的多任务深度学习模型，用于云属性检索。该模型考虑了云属性检索任务之间的层级关系，并具有对不同传感器数据集具有健壮泛化能力的特点。 |
| [^8] | [PhotoBot: Reference-Guided Interactive Photography via Natural Language.](http://arxiv.org/abs/2401.11061) | PhotoBot是一个通过自然语言引导和机器人摄影师相互作用的自动化照片获取框架。它利用视觉语言模型和物体检测器来提供摄影建议，并通过视觉变换器计算相机的姿态调整，从而实现高质量的照片获取。 |
| [^9] | [Artwork Protection Against Neural Style Transfer Using Locally Adaptive Adversarial Color Attack.](http://arxiv.org/abs/2401.09673) | 本文提出了一种名为本地自适应对抗颜色攻击（LAACA）的方法，用于保护艺术品免受神经风格转换（NST）的滥用。该方法通过在不可察觉的情况下对图像进行修改，产生对NST具有干扰作用的扰动。 |
| [^10] | [FUTURE-AI: International consensus guideline for trustworthy and deployable artificial intelligence in healthcare.](http://arxiv.org/abs/2309.12325) | FUTURE-AI是第一个国际共识框架，为医疗保健领域的可信AI工具开发和部署提供指导原则和最佳实践。 |
| [^11] | [Entropy-based Guidance of Deep Neural Networks for Accelerated Convergence and Improved Performance.](http://arxiv.org/abs/2308.14938) | 本研究通过引入基于熵的损失项，通过测量神经网络处理数据时的熵变化，指导神经网络以更快速的收敛、更好的性能学习丰富的潜在数据表示。 |
| [^12] | [LEGO: Learning and Graph-Optimized Modular Tracker for Online Multi-Object Tracking with Point Clouds.](http://arxiv.org/abs/2308.09908) | 本文提出了一个学习和图优化的模块化跟踪器LEGO，通过集成图优化和自注意力机制，提高了在线多目标跟踪中的数据关联性能。使用LiDAR单独进行跟踪的LEGO方法在KITTI目标跟踪评估中表现出了优秀的性能。 |
| [^13] | [Out-of-distribution forgetting: vulnerability of continual learning to intra-class distribution shift.](http://arxiv.org/abs/2306.00427) | 连续学习中存在一种特殊形式的灾难性遗忘——越界遗忘，当给定类别引入类内分布转移时，它会显着削弱该类别的连续学习方法的识别准确率。 |

# 详细

[^1]: 用于低成本心脏血液动力学不稳定性检测的多模态变分自编码器

    Multimodal Variational Autoencoder for Low-cost Cardiac Hemodynamics Instability Detection

    [https://arxiv.org/abs/2403.13658](https://arxiv.org/abs/2403.13658)

    提出了一种新颖的多模态变分自编码器（$\text{CardioVAE}_\text{X,G}$），将低成本胸部X射线（CXR）和心电图（ECG）数据形式整合起来，并实现了共享特征和独特特征的学习。

    

    最近在非侵入性检测心脏血液动力学不稳定性（CHDI）方面取得了进展，主要集中在将机器学习技术应用于单一数据形式，如心脏磁共振成像（MRI）。尽管这些方法具有潜力，但在标记的患者数据量有限时，这些方法通常效果不佳，这是医学领域的常见挑战。此外，只有少数研究探讨了多模态方法来研究CHDI，这些方法主要依赖昂贵的数据形式，如心脏MRI和心脏超声图。为了应对这些限制，我们提出了一种新颖的多模态变分自编码器（$\text{CardioVAE}_\text{X,G}$）来整合低成本胸部X射线（CXR）和心电图（ECG）数据形式，并在大型未标记数据集上进行预训练。具体来说，$\text{CardioVAE}_\text{X,G}$引入了一种新颖的三流预训练策略，以学习共享特征和各数据形式独有的特征，从而实现了fi

    arXiv:2403.13658v1 Announce Type: new  Abstract: Recent advancements in non-invasive detection of cardiac hemodynamic instability (CHDI) primarily focus on applying machine learning techniques to a single data modality, e.g. cardiac magnetic resonance imaging (MRI). Despite their potential, these approaches often fall short especially when the size of labeled patient data is limited, a common challenge in the medical domain. Furthermore, only a few studies have explored multimodal methods to study CHDI, which mostly rely on costly modalities such as cardiac MRI and echocardiogram. In response to these limitations, we propose a novel multimodal variational autoencoder ($\text{CardioVAE}_\text{X,G}$) to integrate low-cost chest X-ray (CXR) and electrocardiogram (ECG) modalities with pre-training on a large unlabeled dataset. Specifically, $\text{CardioVAE}_\text{X,G}$ introduces a novel tri-stream pre-training strategy to learn both shared and modality-specific features, thus enabling fi
    
[^2]: 分层高斯混合正规化流建模用于统一异常检测

    Hierarchical Gaussian Mixture Normalizing Flow Modeling for Unified Anomaly Detection

    [https://arxiv.org/abs/2403.13349](https://arxiv.org/abs/2403.13349)

    提出了一种用于统一异常检测的Hierarchical Gaussian mixture normalizing flow (HGAD)建模方法，通过分层高斯混合建模来提升异常检测模型的表示能力

    

    统一异常检测是异常检测中最具挑战性的任务之一，其中一个统一模型使用来自多个类别的正常样本进行训练，其目标是检测这些类别中的异常。本文提出了一种新颖的分层高斯混合正规化流建模方法，命名为HGAD，用于完成统一异常检测。我们的HGAD包含两个关键组件：跨类别高斯混合建模和类内混合类中心学习。与先前基于NF的AD方法相比，分层高斯混合建模方法可以为异常检测模型带来更强大的表示能力。

    arXiv:2403.13349v1 Announce Type: new  Abstract: Unified anomaly detection (AD) is one of the most challenges for anomaly detection, where one unified model is trained with normal samples from multiple classes with the objective to detect anomalies in these classes. For such a challenging task, popular normalizing flow (NF) based AD methods may fall into a "homogeneous mapping" issue,where the NF-based AD models are biased to generate similar latent representations for both normal and abnormal features, and thereby lead to a high missing rate of anomalies. In this paper, we propose a novel Hierarchical Gaussian mixture normalizing flow modeling method for accomplishing unified Anomaly Detection, which we call HGAD. Our HGAD consists of two key components: inter-class Gaussian mixture modeling and intra-class mixed class centers learning. Compared to the previous NF-based AD methods, the hierarchical Gaussian mixture modeling approach can bring stronger representation capability to the 
    
[^3]: DexCap：用于灵巧操作的可扩展和可移植动作捕捉数据收集系统

    DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation

    [https://arxiv.org/abs/2403.07788](https://arxiv.org/abs/2403.07788)

    DexCap是一个可移植的手部动作捕捉系统，结合DexIL算法从人类手部运动数据中训练机器人技能，具有精确追踪和复制人类动作的能力。

    

    从人类手部运动数据中学习是为机器人赋予类人灵巧在现实操纵任务中的潜在途径，然而，现存手部动作捕捉系统的可移植性以及将动作捕捉数据转化为有效控制策略的困难仍然存在挑战。为了应对这些问题，我们引入了DexCap，一个便携式手部动作捕捉系统，以及DexIL，一种新颖的模仿算法，可直接从人类手部动作捕捉数据训练灵巧机器人技能。DexCap基于SLAM和电磁场以及环境的3D观察，提供了对手腕和手指运动的精确、抗遮挡的跟踪。利用这一丰富的数据集，DexIL采用逆运动学和基于点云的模仿学习来复制人类动作与机器人手。除了从人类运动中学习外，DexCap还提供了一种op

    arXiv:2403.07788v1 Announce Type: cross  Abstract: Imitation learning from human hand motion data presents a promising avenue for imbuing robots with human-like dexterity in real-world manipulation tasks. Despite this potential, substantial challenges persist, particularly with the portability of existing hand motion capture (mocap) systems and the difficulty of translating mocap data into effective control policies. To tackle these issues, we introduce DexCap, a portable hand motion capture system, alongside DexIL, a novel imitation algorithm for training dexterous robot skills directly from human hand mocap data. DexCap offers precise, occlusion-resistant tracking of wrist and finger motions based on SLAM and electromagnetic field together with 3D observations of the environment. Utilizing this rich dataset, DexIL employs inverse kinematics and point cloud-based imitation learning to replicate human actions with robot hands. Beyond learning from human motion, DexCap also offers an op
    
[^4]: 通过偏差净化实现多模态情感分析的研究

    Towards Multimodal Sentiment Analysis Debiasing via Bias Purification

    [https://arxiv.org/abs/2403.05023](https://arxiv.org/abs/2403.05023)

    提出了一种基于因果关系的多模态对事实推理情感分析框架，用于净化和缓解数据集的偏见，从而提高多模态情感分析的性能。

    

    多模态情感分析（MSA）旨在通过整合来自不同模态（如视觉、语言和音频）的与情感相关线索来理解人类意图。然而，当前MSA任务普遍受到未经计划的数据集偏见的影响，尤其是多模态话语级标签偏见和单词级上下文偏见。这些有害的偏见可能会误导模型专注于统计捷径和错误相关性，导致严重的性能瓶颈。为了缓解这些问题，我们提出了一种基于因果关系而非传统似然性的多模态对事实推理情感（MCIS）分析框架。具体而言，我们首先制定一个因果图来发现已训练的基准模型中的有害偏见。在推理阶段，给定一个事实多模态输入，MCIS想象两种对事实情形，以净化和缓解这些偏见。然后，MCIS可以从偏差中做出不带偏见的决策。

    arXiv:2403.05023v1 Announce Type: new  Abstract: Multimodal Sentiment Analysis (MSA) aims to understand human intentions by integrating emotion-related clues from diverse modalities, such as visual, language, and audio. Unfortunately, the current MSA task invariably suffers from unplanned dataset biases, particularly multimodal utterance-level label bias and word-level context bias. These harmful biases potentially mislead models to focus on statistical shortcuts and spurious correlations, causing severe performance bottlenecks. To alleviate these issues, we present a Multimodal Counterfactual Inference Sentiment (MCIS) analysis framework based on causality rather than conventional likelihood. Concretely, we first formulate a causal graph to discover harmful biases from already-trained vanilla models. In the inference phase, given a factual multimodal input, MCIS imagines two counterfactual scenarios to purify and mitigate these biases. Then, MCIS can make unbiased decisions from biase
    
[^5]: U$^2$MRPD: 通过大型潜在扩散模型引导的无监督MRI重建

    U$^2$MRPD: Unsupervised undersampled MRI reconstruction by prompting a large latent diffusion model

    [https://arxiv.org/abs/2402.10609](https://arxiv.org/abs/2402.10609)

    U$^2$MRPD是一个新颖的框架，通过大型潜在扩散模型引导，实现了无监督的欠采样MRI重建，能够支持图像特定的MRI重建，且在多个数据集上表现出与监督和MRI扩散方法相媲美甚至更好的性能。

    

    arXiv:2402.10609v1 公告类型: 跨领域 摘要: 在自然图像上预训练的大型潜在扩散模型(LLDM)中蕴含着丰富而假设上普遍适用于自然和医学图像的隐含视觉知识。为了测试这一假设，我们引入了一个新颖的框架，通过提示一个预训练的大型潜在扩散模型（U$^2$MRPD）进行无监督的欠采样MRI重建。现有的数据驱动、监督的欠采样MRI重建网络通常具有有限的泛化能力和适应性，不足以应对各种数据采集场景；然而，U$^2$MRPD通过使用量身定制的MRSampler，支持图像特定的MRI重建，该MRSampler适用于复值MRI图像。通过任何单一来源或多源MRI数据集，U$^2$MRPD的性能还可以通过MRAdapter进行进一步提升，同时保持生成图像先验不变。多个数据集上的实验表明，U$^2$MRPD实现了与监督和MRI扩散方法相媲美甚至更好的性能。

    arXiv:2402.10609v1 Announce Type: cross  Abstract: Implicit visual knowledge in a large latent diffusion model (LLDM) pre-trained on natural images is rich and hypothetically universal to natural and medical images. To test this hypothesis, we introduce a novel framework for Unsupervised Undersampled MRI Reconstruction by Prompting a pre-trained large latent Diffusion model ( U$^2$MRPD). Existing data-driven, supervised undersampled MRI reconstruction networks are typically of limited generalizability and adaptability toward diverse data acquisition scenarios; yet U$^2$MRPD supports image-specific MRI reconstruction by prompting an LLDM with an MRSampler tailored for complex-valued MRI images. With any single-source or diverse-source MRI dataset, U$^2$MRPD's performance is further boosted by an MRAdapter while keeping the generative image priors intact. Experiments on multiple datasets show that U$^2$MRPD achieves comparable or better performance than supervised and MRI diffusion metho
    
[^6]: ScreenAI: 用于UI和信息图表理解的视觉-语言模型

    ScreenAI: A Vision-Language Model for UI and Infographics Understanding

    [https://arxiv.org/abs/2402.04615](https://arxiv.org/abs/2402.04615)

    ScreenAI是一个专注于UI和信息图表理解的视觉-语言模型，通过灵活的修补策略和独特的数据集训练，以及针对UI元素的屏幕注解任务的处理，实现了在多个任务上的新的最优结果。

    

    屏幕用户界面（UI）和信息图表在人类沟通和人机交互中起着重要作用，并且共享相似的视觉语言和设计原则。我们介绍了ScreenAI，这是一个专门用于UI和信息图表理解的视觉-语言模型。我们的模型改进了PaLI架构，采用了pix2struct的灵活修补策略，并经过独特的数据集训练。在这个数据集的核心是一项新颖的屏幕注解任务，模型必须识别UI元素的类型和位置。我们使用这些文本注解来描述屏幕，并使用大规模的语言模型自动生成问答（QA），UI导航和摘要训练数据集。我们进行了消融研究以展示这些设计选择的影响。在仅有5B参数的情况下，ScreenAI在基于UI和信息图表的任务（多页文档VQA，WebSRC，MoTIF和Widget字幕）上取得了最新的最优结果，并且达到了最好的效果。

    Screen user interfaces (UIs) and infographics, sharing similar visual language and design principles, play important roles in human communication and human-machine interaction. We introduce ScreenAI, a vision-language model that specializes in UI and infographics understanding. Our model improves upon the PaLI architecture with the flexible patching strategy of pix2struct and is trained on a unique mixture of datasets. At the heart of this mixture is a novel screen annotation task in which the model has to identify the type and location of UI elements. We use these text annotations to describe screens to Large Language Models and automatically generate question-answering (QA), UI navigation, and summarization training datasets at scale. We run ablation studies to demonstrate the impact of these design choices. At only 5B parameters, ScreenAI achieves new state-of-the-artresults on UI- and infographics-based tasks (Multi-page DocVQA, WebSRC, MoTIF and Widget Captioning), and new best-in
    
[^7]: MT-HCCAR: 多任务深度学习与层级分类的注意力回归用于云属性检索

    MT-HCCAR: Multi-Task Deep Learning with Hierarchical Classification and Attention-based Regression for Cloud Property Retrieval. (arXiv:2401.16520v1 [cs.LG])

    [http://arxiv.org/abs/2401.16520](http://arxiv.org/abs/2401.16520)

    这篇论文提出了一种名为MT-HCCAR的多任务深度学习模型，用于云属性检索。该模型考虑了云属性检索任务之间的层级关系，并具有对不同传感器数据集具有健壮泛化能力的特点。

    

    在地球科学领域中，有效的云属性检索包括云遮蔽、云相分类和云光学厚度（COT）预测仍然至关重要。传统方法需要针对每个传感器仪器使用不同的模型，因为它们具有独特的光谱特征。最近，在地球科学研究中采用了机器学习和深度学习技术从卫星数据集的光谱观测中提取特征。然而，现有方法缺乏考虑检索任务之间层级关系的创新架构。此外，考虑到现有传感器之间的光谱多样性，开发具有对不同传感器数据集具有健壮泛化能力的模型是必要的。令人惊讶的是，目前缺乏解决多样数据集下选择最优模型的方法。为此，本文引入了MT-HCCAR，这是一种端到端的深度学习模型，采用多任务学习和基于注意力的回归方法。

    In the realm of Earth science, effective cloud property retrieval, encompassing cloud masking, cloud phase classification, and cloud optical thickness (COT) prediction, remains pivotal. Traditional methodologies necessitate distinct models for each sensor instrument due to their unique spectral characteristics. Recent strides in Earth Science research have embraced machine learning and deep learning techniques to extract features from satellite datasets' spectral observations. However, prevailing approaches lack novel architectures accounting for hierarchical relationships among retrieval tasks. Moreover, considering the spectral diversity among existing sensors, the development of models with robust generalization capabilities over different sensor datasets is imperative. Surprisingly, there is a dearth of methodologies addressing the selection of an optimal model for diverse datasets. In response, this paper introduces MT-HCCAR, an end-to-end deep learning model employing multi-task 
    
[^8]: PhotoBot：通过自然语言引导的参考互动摄影

    PhotoBot: Reference-Guided Interactive Photography via Natural Language. (arXiv:2401.11061v1 [cs.CV])

    [http://arxiv.org/abs/2401.11061](http://arxiv.org/abs/2401.11061)

    PhotoBot是一个通过自然语言引导和机器人摄影师相互作用的自动化照片获取框架。它利用视觉语言模型和物体检测器来提供摄影建议，并通过视觉变换器计算相机的姿态调整，从而实现高质量的照片获取。

    

    我们介绍了一个名为PhotoBot的框架，它基于高级人类语言引导和机器人摄影师之间的相互作用，用于自动化的照片获取。我们建议通过从策展画廊中检索到的参考图片向用户传达摄影建议。我们利用视觉语言模型（VLM）和物体检测器，通过文本描述对参考图片进行特征化，并使用大型语言模型（LLM）通过基于用户语言查询的文本推理检索相关的参考图片。为了对应参考图片和观察到的场景，我们利用一个能够捕捉显著不同的图像的语义相似性的预训练特征的视觉变换器，通过解决一个透视n-点（PnP）问题来计算RGB-D相机的姿态调整。我们在配备有手腕相机的真实机械手臂上演示了我们的方法。我们的用户研究表明，由PhotoBot拍摄的照片具有良好的质量和效果。

    We introduce PhotoBot, a framework for automated photo acquisition based on an interplay between high-level human language guidance and a robot photographer. We propose to communicate photography suggestions to the user via a reference picture that is retrieved from a curated gallery. We exploit a visual language model (VLM) and an object detector to characterize reference pictures via textual descriptions and use a large language model (LLM) to retrieve relevant reference pictures based on a user's language query through text-based reasoning. To correspond the reference picture and the observed scene, we exploit pre-trained features from a vision transformer capable of capturing semantic similarity across significantly varying images. Using these features, we compute pose adjustments for an RGB-D camera by solving a Perspective-n-Point (PnP) problem. We demonstrate our approach on a real-world manipulator equipped with a wrist camera. Our user studies show that photos taken by PhotoBo
    
[^9]: 使用本地自适应对抗颜色攻击对艺术品进行神经风格转换的保护

    Artwork Protection Against Neural Style Transfer Using Locally Adaptive Adversarial Color Attack. (arXiv:2401.09673v1 [cs.CV])

    [http://arxiv.org/abs/2401.09673](http://arxiv.org/abs/2401.09673)

    本文提出了一种名为本地自适应对抗颜色攻击（LAACA）的方法，用于保护艺术品免受神经风格转换（NST）的滥用。该方法通过在不可察觉的情况下对图像进行修改，产生对NST具有干扰作用的扰动。

    

    神经风格转换（NST）广泛应用于计算机视觉中，用于生成具有任意风格的新图像。这个过程利用神经网络将风格图像的美学元素与内容图像的结构因素融合在一起，形成一个和谐整合的视觉结果。然而，未经授权的NST可能会滥用艺术品。这种滥用引起了关于艺术家权利的社会技术问题，并促使开发技术方法来积极保护原始创作。对抗性攻击主要在机器学习安全中进行探索。我们的工作将这一技术引入到保护艺术家知识产权的领域。本文引入了本地自适应对抗颜色攻击（LAACA）的方法，这种方法可以以对人眼不可察觉但对NST产生干扰的方式修改图像。具体而言，我们设计了针对高频内容丰富区域的扰动，这些扰动由中间特征的破坏产生。我们进行了实验和用户研究。

    Neural style transfer (NST) is widely adopted in computer vision to generate new images with arbitrary styles. This process leverages neural networks to merge aesthetic elements of a style image with the structural aspects of a content image into a harmoniously integrated visual result. However, unauthorized NST can exploit artwork. Such misuse raises socio-technical concerns regarding artists' rights and motivates the development of technical approaches for the proactive protection of original creations. Adversarial attack is a concept primarily explored in machine learning security. Our work introduces this technique to protect artists' intellectual property. In this paper Locally Adaptive Adversarial Color Attack (LAACA), a method for altering images in a manner imperceptible to the human eyes but disruptive to NST. Specifically, we design perturbations targeting image areas rich in high-frequency content, generated by disrupting intermediate features. Our experiments and user study
    
[^10]: FUTURE-AI：在医疗保健领域的可信和可部署人工智能的国际共识指南

    FUTURE-AI: International consensus guideline for trustworthy and deployable artificial intelligence in healthcare. (arXiv:2309.12325v1 [cs.CY])

    [http://arxiv.org/abs/2309.12325](http://arxiv.org/abs/2309.12325)

    FUTURE-AI是第一个国际共识框架，为医疗保健领域的可信AI工具开发和部署提供指导原则和最佳实践。

    

    尽管在医学和医疗保健领域人工智能（AI）取得了重大进展，但AI技术在现实临床实践中的部署和采用仍受限。近年来，人们对医疗AI的技术、临床、伦理和法律风险提出了关注。为了增加在现实世界中的采用，医疗AI工具必须得到患者、临床医生、健康组织和当局的信任和接受。本文描述了FUTURE-AI指南作为第一个用于指导医疗保健领域可信AI工具开发和部署的国际共识框架。FUTURE-AI联盟成立于2021年，目前包括来自51个国家的118位跨学科专家，代表了所有大洲，包括AI科学家、临床医生、伦理学家和社会科学家。在为期两年的时间里，联盟通过迭代过程定义了可信AI的指导原则和最佳实践，其中包括

    Despite major advances in artificial intelligence (AI) for medicine and healthcare, the deployment and adoption of AI technologies remain limited in real-world clinical practice. In recent years, concerns have been raised about the technical, clinical, ethical and legal risks associated with medical AI. To increase real world adoption, it is essential that medical AI tools are trusted and accepted by patients, clinicians, health organisations and authorities. This work describes the FUTURE-AI guideline as the first international consensus framework for guiding the development and deployment of trustworthy AI tools in healthcare. The FUTURE-AI consortium was founded in 2021 and currently comprises 118 inter-disciplinary experts from 51 countries representing all continents, including AI scientists, clinicians, ethicists, and social scientists. Over a two-year period, the consortium defined guiding principles and best practices for trustworthy AI through an iterative process comprising a
    
[^11]: 基于熵的指导深度神经网络加速收敛和改善性能

    Entropy-based Guidance of Deep Neural Networks for Accelerated Convergence and Improved Performance. (arXiv:2308.14938v1 [cs.CV])

    [http://arxiv.org/abs/2308.14938](http://arxiv.org/abs/2308.14938)

    本研究通过引入基于熵的损失项，通过测量神经网络处理数据时的熵变化，指导神经网络以更快速的收敛、更好的性能学习丰富的潜在数据表示。

    

    神经网络极大地增加了我们从大规模、高维度数据集中学习的能力，跨越无数学科。然而，它们的决策不易解释，计算成本高，建立和训练它们是不确定的过程。为了给这些努力增加结构，我们推导出了新的数学结果，以高效地测量全连接和卷积神经网络处理数据时的熵变化，并引入了基于熵的损失项。在基准数据集上进行的图像压缩和图像分类实验表明，这些损失项指导神经网络以更少的维度学习丰富的潜在数据表示，收敛于更少的训练轮次，并取得更好的测试指标。

    Neural networks have dramatically increased our capacity to learn from large, high-dimensional datasets across innumerable disciplines. However, their decisions are not easily interpretable, their computational costs are high, and building and training them are uncertain processes. To add structure to these efforts, we derive new mathematical results to efficiently measure the changes in entropy as fully-connected and convolutional neural networks process data, and introduce entropy-based loss terms. Experiments in image compression and image classification on benchmark datasets demonstrate these losses guide neural networks to learn rich latent data representations in fewer dimensions, converge in fewer training epochs, and achieve better test metrics.
    
[^12]: LEGO: 对于基于点云的在线多目标跟踪的学习和图优化的模块化跟踪器

    LEGO: Learning and Graph-Optimized Modular Tracker for Online Multi-Object Tracking with Point Clouds. (arXiv:2308.09908v1 [cs.CV])

    [http://arxiv.org/abs/2308.09908](http://arxiv.org/abs/2308.09908)

    本文提出了一个学习和图优化的模块化跟踪器LEGO，通过集成图优化和自注意力机制，提高了在线多目标跟踪中的数据关联性能。使用LiDAR单独进行跟踪的LEGO方法在KITTI目标跟踪评估中表现出了优秀的性能。

    

    在线多目标跟踪（MOT）在自主系统中起着关键作用。现有的最先进方法通常采用跟踪-检测方法，数据关联起到了至关重要的作用。本文提出了一个学习和图优化（LEGO）的模块化跟踪器，以提高数据关联性能。所提出的LEGO跟踪器集成了图优化和自注意力机制，能够有效地制定关联评分图，从而实现准确高效的目标匹配。为了进一步增强状态更新过程，本文还添加了卡尔曼滤波器，通过将对象状态的时间连贯性纳入跟踪中，确保一致的跟踪。与其他在线跟踪方法（包括基于LiDAR和基于LiDAR-相机融合的方法）相比，我们提出的仅利用LiDAR的方法表现出了卓越性能。在提交结果至KITTI目标跟踪评估排行榜时，LEGO排名第一。

    Online multi-object tracking (MOT) plays a pivotal role in autonomous systems. The state-of-the-art approaches usually employ a tracking-by-detection method, and data association plays a critical role. This paper proposes a learning and graph-optimized (LEGO) modular tracker to improve data association performance in the existing literature. The proposed LEGO tracker integrates graph optimization and self-attention mechanisms, which efficiently formulate the association score map, facilitating the accurate and efficient matching of objects across time frames. To further enhance the state update process, the Kalman filter is added to ensure consistent tracking by incorporating temporal coherence in the object states. Our proposed method utilizing LiDAR alone has shown exceptional performance compared to other online tracking approaches, including LiDAR-based and LiDAR-camera fusion-based methods. LEGO ranked 1st at the time of submitting results to KITTI object tracking evaluation ranki
    
[^13]: 针对类内分布转移的过度遗忘：连续学习的脆弱性

    Out-of-distribution forgetting: vulnerability of continual learning to intra-class distribution shift. (arXiv:2306.00427v1 [cs.LG])

    [http://arxiv.org/abs/2306.00427](http://arxiv.org/abs/2306.00427)

    连续学习中存在一种特殊形式的灾难性遗忘——越界遗忘，当给定类别引入类内分布转移时，它会显着削弱该类别的连续学习方法的识别准确率。

    

    连续学习是让人工神经网络在开放环境中工作的重要技术。在联合学习中，人们已经知道由意图攻击或环境扰动引起的越界问题严重影响网络的泛化能力。在这项工作中，我们报告了连续学习设置中由越界问题引起的一种特殊形式的灾难性遗忘，我们将其称为越界遗忘（OODF）。在连续图像分类任务中，我们发现，针对给定类别，引入类内分布转移显着削弱了后续学习过程中该类别的连续学习方法的识别准确率。有趣的是，这种现象对于连续学习而言是特殊的，因为同样级别的分布转移只有微不足道的影响。

    Continual learning (CL) is an important technique to allow artificial neural networks to work in open environments. CL enables a system to learn new tasks without severe interference to its performance on old tasks, i.e., overcome the problems of catastrophic forgetting. In joint learning, it is well known that the out-of-distribution (OOD) problem caused by intentional attacks or environmental perturbations will severely impair the ability of networks to generalize. In this work, we reported a special form of catastrophic forgetting raised by the OOD problem in continual learning settings, and we named it out-of-distribution forgetting (OODF). In continual image classification tasks, we found that for a given category, introducing an intra-class distribution shift significantly impaired the recognition accuracy of CL methods for that category during subsequent learning. Interestingly, this phenomenon is special for CL as the same level of distribution shift had only negligible effects
    

