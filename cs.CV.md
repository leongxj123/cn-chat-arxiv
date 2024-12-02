# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Object-Centric Domain Randomization for 3D Shape Reconstruction in the Wild](https://arxiv.org/abs/2403.14539) | 提出了ObjectDR，利用对象-centric的域随机化合成单视图3D形状重建中缺乏的配对数据，通过条件生成模型和解耦框架来生成和保留对象轮廓以及广泛变化的数据，从而为培训模型捕捉域不变性几何形状。 |
| [^2] | [Efficient Prompt Tuning of Large Vision-Language Model for Fine-Grained Ship Classification](https://arxiv.org/abs/2403.08271) | 本研究使用大规模预训练的视觉-语言模型，提出了一种高效的提示调整方法，以增强未见船舶类别的分类准确性。 |
| [^3] | [Asclepius: A Spectrum Evaluation Benchmark for Medical Multi-Modal Large Language Models](https://arxiv.org/abs/2402.11217) | Asclepius是一个新的医学多模态大语言模型基准，旨在为可信的Med-MLLMs评估提供单独且临床代表性的评估方案。 |
| [^4] | [Diffusion MRI with Machine Learning](https://arxiv.org/abs/2402.00019) | 本文评估了机器学习在弥散磁共振成像中的应用，重点关注了微结构映射、纤维束描记、白质纤维束分析以及数据预处理和协调的方法。通过对现有方法的总结，提出了未来研究的主题。 |
| [^5] | [An explainable three dimension framework to uncover learning patterns: A unified look in variable sulci recognition](https://arxiv.org/abs/2309.00903) | 该论文提出了一个针对医学成像中的可解释AI的三维框架，旨在解决神经科学领域中识别大脑沟特征的复杂性问题。 |
| [^6] | [A Bayesian Unification of Self-Supervised Clustering and Energy-Based Models.](http://arxiv.org/abs/2401.00873) | 该论文研究了用贝叶斯方法统一自监督聚类和能量模型，提出了一种标准化的推导方法，并设计了一个新的可靠地惩罚失败模式的下界。这个下界使得能够训练一个标准的骨架架构，而无需使用非对称元素。 |
| [^7] | [A Survey on Multimodal Large Language Models.](http://arxiv.org/abs/2306.13549) | 本文追踪和总结了多模态大语言模型（MLLM）的最新进展，包括多模态指令调整、多模态上下文学习、多模态思维链和LLM辅助视觉推理等应用，指出了现有挑战和有前途的研究方向。 |
| [^8] | [DDT: A Diffusion-Driven Transformer-based Framework for Human Mesh Recovery from a Video.](http://arxiv.org/abs/2303.13397) | 提出了一种基于扩散驱动变压器的视频 HMR 框架（DDT），它旨在从输入序列中解码特定的运动模式，增强运动平滑性和时间一致性，并输出所有帧的人体网格，使得 DDT 更适用于时间效率至关重要的实际应用。 |

# 详细

[^1]: Object-Centric Domain Randomization用于野外3D形状重建

    Object-Centric Domain Randomization for 3D Shape Reconstruction in the Wild

    [https://arxiv.org/abs/2403.14539](https://arxiv.org/abs/2403.14539)

    提出了ObjectDR，利用对象-centric的域随机化合成单视图3D形状重建中缺乏的配对数据，通过条件生成模型和解耦框架来生成和保留对象轮廓以及广泛变化的数据，从而为培训模型捕捉域不变性几何形状。

    

    单视图3D形状在野外的重建面临的最大挑战之一是来自真实环境中的<3D形状，2D图像>-配对数据的稀缺性。受域随机化引人注目的成就的启发，我们提出了ObjectDR，通过对对象外观和背景的视觉变化进行随机仿真，合成这种配对数据。我们的数据合成框架利用条件生成模型（例如ControlNet）生成符合空间条件（例如2.5D草图）的图像，这些条件可以通过从对象集合（例如Objaverse-XL）的渲染过程获得3D形状。为了模拟多样化的变化同时保留嵌入空间条件中的对象轮廓，我们还引入了一个利用初始对象指导的解耦框架。

    arXiv:2403.14539v1 Announce Type: cross  Abstract: One of the biggest challenges in single-view 3D shape reconstruction in the wild is the scarcity of <3D shape, 2D image>-paired data from real-world environments. Inspired by remarkable achievements via domain randomization, we propose ObjectDR which synthesizes such paired data via a random simulation of visual variations in object appearances and backgrounds. Our data synthesis framework exploits a conditional generative model (e.g., ControlNet) to generate images conforming to spatial conditions such as 2.5D sketches, which are obtainable through a rendering process of 3D shapes from object collections (e.g., Objaverse-XL). To simulate diverse variations while preserving object silhouettes embedded in spatial conditions, we also introduce a disentangled framework which leverages an initial object guidance. After synthesizing a wide range of data, we pre-train a model on them so that it learns to capture a domain-invariant geometry p
    
[^2]: 用于细粒度船舶分类的大规模视觉-语言模型的高效提示调整

    Efficient Prompt Tuning of Large Vision-Language Model for Fine-Grained Ship Classification

    [https://arxiv.org/abs/2403.08271](https://arxiv.org/abs/2403.08271)

    本研究使用大规模预训练的视觉-语言模型，提出了一种高效的提示调整方法，以增强未见船舶类别的分类准确性。

    

    遥感中的细粒度船舶分类 (RS-FGSC) 由于类别之间的高相似性以及有限的标记数据可用性而面临重大挑战，限制了传统监督分类方法的有效性。最近大规模预训练的视觉-语言模型 (VLMs) 在少样本或零样本学习中展现出令人印象深刻的能力，特别是在理解图像内容方面。本研究深入挖掘了VLMs的潜力，以提高未见船舶类别的分类准确性，在由于成本或隐私限制而数据受限的情况下具有重要意义。直接为RS-FGSC微调VLMs通常会遇到过拟合可见类的挑战，导致对未见类的泛化不佳，突出了区分复杂背景和捕捉独特船舶特征的困难。

    arXiv:2403.08271v1 Announce Type: cross  Abstract: Fine-grained ship classification in remote sensing (RS-FGSC) poses a significant challenge due to the high similarity between classes and the limited availability of labeled data, limiting the effectiveness of traditional supervised classification methods. Recent advancements in large pre-trained Vision-Language Models (VLMs) have demonstrated impressive capabilities in few-shot or zero-shot learning, particularly in understanding image content. This study delves into harnessing the potential of VLMs to enhance classification accuracy for unseen ship categories, which holds considerable significance in scenarios with restricted data due to cost or privacy constraints. Directly fine-tuning VLMs for RS-FGSC often encounters the challenge of overfitting the seen classes, resulting in suboptimal generalization to unseen classes, which highlights the difficulty in differentiating complex backgrounds and capturing distinct ship features. To 
    
[^3]: Asclepius：用于医学多模态大语言模型的频谱评估基准

    Asclepius: A Spectrum Evaluation Benchmark for Medical Multi-Modal Large Language Models

    [https://arxiv.org/abs/2402.11217](https://arxiv.org/abs/2402.11217)

    Asclepius是一个新的医学多模态大语言模型基准，旨在为可信的Med-MLLMs评估提供单独且临床代表性的评估方案。

    

    arXiv:2402.11217v1 公告类型：新摘要：医学多模态大语言模型（Med-MLLMs）的重大突破通过强大的信息综合和医疗决策支持改造了现代医疗保健。然而，由于现实世界诊断框架的复杂性涵盖了各种医学专业，并涉及复杂的临床决策，这些模型通常在不适合Med-MLLMs的基准上进行评估。此外，由于Med-MLLMs是在大量公开可用数据集上进行训练的，这些基准容易出现数据泄露。因此，需要一个独立且临床代表性的基准用于可信的Med-MLLMs评估。为此，我们引入了Asclepius，一个新颖的Med-MLLM基准，严格和全面评估模型在不同医学专业（心血管、胃肠等）和不同诊断能力（知觉、疾病分析等）方面的能力。

    arXiv:2402.11217v1 Announce Type: new  Abstract: The significant breakthroughs of Medical Multi-Modal Large Language Models (Med-MLLMs) renovate modern healthcare with robust information synthesis and medical decision support. However, these models are often evaluated on benchmarks that are unsuitable for the Med-MLLMs due to the intricate nature of the real-world diagnostic frameworks, which encompass diverse medical specialties and involve complex clinical decisions. Moreover, these benchmarks are susceptible to data leakage, since Med-MLLMs are trained on large assemblies of publicly available data. Thus, an isolated and clinically representative benchmark is highly desirable for credible Med-MLLMs evaluation. To this end, we introduce Asclepius, a novel Med-MLLM benchmark that rigorously and comprehensively assesses model capability in terms of: distinct medical specialties (cardiovascular, gastroenterology, etc.) and different diagnostic capacities (perception, disease analysis, e
    
[^4]: 机器学习在弥散磁共振成像中的应用

    Diffusion MRI with Machine Learning

    [https://arxiv.org/abs/2402.00019](https://arxiv.org/abs/2402.00019)

    本文评估了机器学习在弥散磁共振成像中的应用，重点关注了微结构映射、纤维束描记、白质纤维束分析以及数据预处理和协调的方法。通过对现有方法的总结，提出了未来研究的主题。

    

    弥散加权磁共振成像（dMRI）具有非侵入性评估大脑微结构和结构连接的独特能力。然而，分析dMRI数据以提取临床和科学目的的有用信息具有挑战性。 dMRI测量通常受到强噪声和伪影的干扰，数据中通常存在高的会话间和扫描者间异质性，以及大脑结构的相当大的个体间变异，并且测量和感兴趣现象之间的关系可能非常复杂。近年来，机器学习方法在dMRI分析中的应用越来越多。本文旨在评估这些尝试，重点关注已经解决了微结构映射、纤维束描记、白质纤维束分析以及数据预处理和协调的方法。我们总结了现有方法的主要发现、优点和缺点，并提出了未来研究的主题。

    Diffusion-weighted magnetic resonance imaging (dMRI) offers unique capabilities such as noninvasive assessment of brain's micro-structure and structural connectivity. However, analyzing the dMRI data to extract useful information for clinical and scientific purposes is challenging. The dMRI measurements often suffer from strong noise and artifacts, there is usually high inter-session and inter-scanner heterogeneity in the data and considerable inter-subject variability in brain structure, and the relationship between measurements and the phenomena of interest can be highly complex. Recent years have witnessed increasing use of machine learning methods for dMRI analysis. This manuscript aims to assess these efforts, with a focus on methods that have addressed micro-structure mapping, tractography, white matter tract analysis, as well as data preprocessing and harmonization. We summarize the main findings, strengths, and weaknesses of the existing methods and suggest topics for future re
    
[^5]: 一种可解释的三维框架揭示学习模式：变量脑沟识别的统一视角

    An explainable three dimension framework to uncover learning patterns: A unified look in variable sulci recognition

    [https://arxiv.org/abs/2309.00903](https://arxiv.org/abs/2309.00903)

    该论文提出了一个针对医学成像中的可解释AI的三维框架，旨在解决神经科学领域中识别大脑沟特征的复杂性问题。

    

    可解释的人工智能在医学成像中至关重要。在挑战性的神经科学领域里，视觉主题在三维空间内表现出高度复杂性。神经科学的应用涉及从MRI中识别大脑沟特征，由于专家之间的标注规程存在差异和大脑复杂的三维功能，我们面临着重大障碍。因此，传统的可解释性方法在有效验证和评估这些网络方面表现不佳。为了解决这个问题，我们首先提出了数学公式，细化了不同计算机视觉任务中解释需求的各种类别，分为自解释、半解释、非解释和基于验证协议可靠性的新模式学习应用。根据这个数学公式，我们提出了一个旨在解释三维的框架。

    arXiv:2309.00903v2 Announce Type: replace-cross  Abstract: Explainable AI is crucial in medical imaging. In the challenging field of neuroscience, visual topics present a high level of complexity, particularly within three-dimensional space. The application of neuroscience, which involves identifying brain sulcal features from MRI, faces significant hurdles due to varying annotation protocols among experts and the intricate three-dimension functionality of the brain. Consequently, traditional explainability approaches fall short in effectively validating and evaluating these networks. To address this, we first present a mathematical formulation delineating various categories of explanation needs across diverse computer vision tasks, categorized into self-explanatory, semi-explanatory, non-explanatory, and new-pattern learning applications based on the reliability of the validation protocol. With respect to this mathematical formulation, we propose a 3D explainability framework aimed at
    
[^6]: 用贝叶斯方法统一自监督聚类和能量模型

    A Bayesian Unification of Self-Supervised Clustering and Energy-Based Models. (arXiv:2401.00873v1 [cs.LG])

    [http://arxiv.org/abs/2401.00873](http://arxiv.org/abs/2401.00873)

    该论文研究了用贝叶斯方法统一自监督聚类和能量模型，提出了一种标准化的推导方法，并设计了一个新的可靠地惩罚失败模式的下界。这个下界使得能够训练一个标准的骨架架构，而无需使用非对称元素。

    

    自监督学习是一种利用大量无标签数据的流行且强大的方法，文献中提出了各种训练目标。本研究对最先进的自监督学习目标进行贝叶斯分析，阐明了每个类别中潜在的概率图模型，并提出了一种从基本原理出发推导这些模型的标准方法。分析还表明了将自监督学习与基于似然的生成模型自然整合的方法。我们在基于聚类的自监督学习和能量模型领域中实现了这个概念，引入了一个新的下界，经证明能可靠地惩罚最重要的失败模式。此外，这个新提出的下界使得能够训练一个标准的骨干架构，而无需使用诸如停止梯度、动量编码器或专门的聚类等非对称元素。

    Self-supervised learning is a popular and powerful method for utilizing large amounts of unlabeled data, for which a wide variety of training objectives have been proposed in the literature. In this study, we perform a Bayesian analysis of state-of-the-art self-supervised learning objectives, elucidating the underlying probabilistic graphical models in each class and presenting a standardized methodology for their derivation from first principles. The analysis also indicates a natural means of integrating self-supervised learning with likelihood-based generative models. We instantiate this concept within the realm of cluster-based self-supervised learning and energy models, introducing a novel lower bound which is proven to reliably penalize the most important failure modes. Furthermore, this newly proposed lower bound enables the training of a standard backbone architecture without the necessity for asymmetric elements such as stop gradients, momentum encoders, or specialized clusteri
    
[^7]: 多模态大语言模型综述

    A Survey on Multimodal Large Language Models. (arXiv:2306.13549v1 [cs.CV])

    [http://arxiv.org/abs/2306.13549](http://arxiv.org/abs/2306.13549)

    本文追踪和总结了多模态大语言模型（MLLM）的最新进展，包括多模态指令调整、多模态上下文学习、多模态思维链和LLM辅助视觉推理等应用，指出了现有挑战和有前途的研究方向。

    

    多模态大语言模型（MLLM）是一种新兴的研究热点，使用强大的大语言模型作为大脑执行多模态任务。MLLM 的惊人能力，如基于图像编写故事和无OCR数学推理等，在传统方法中很少见，表明了通向人工智能的潜在路径。本文旨在追踪和总结 MLLM 的最新进展。首先，我们介绍了 MLLM 的构成，概述了相关概念。然后，讨论了关键技术和应用，包括多模态指令调整（M-IT）、多模态上下文学习（M-ICL）、多模态思维链（M-CoT）和LLM辅助视觉推理（LAVR）。最后，我们讨论了现有的挑战，并指出了有前途的研究方向。鉴于 MLLM 时代才刚刚开始，我们会不断更新这个综述，并希望能激发更多的研究。

    Multimodal Large Language Model (MLLM) recently has been a new rising research hotspot, which uses powerful Large Language Models (LLMs) as a brain to perform multimodal tasks. The surprising emergent capabilities of MLLM, such as writing stories based on images and OCR-free math reasoning, are rare in traditional methods, suggesting a potential path to artificial general intelligence. In this paper, we aim to trace and summarize the recent progress of MLLM. First of all, we present the formulation of MLLM and delineate its related concepts. Then, we discuss the key techniques and applications, including Multimodal Instruction Tuning (M-IT), Multimodal In-Context Learning (M-ICL), Multimodal Chain of Thought (M-CoT), and LLM-Aided Visual Reasoning (LAVR). Finally, we discuss existing challenges and point out promising research directions. In light of the fact that the era of MLLM has only just begun, we will keep updating this survey and hope it can inspire more research. An associated
    
[^8]: DDT：一种基于扩散驱动变压器的从视频中恢复人体网格的框架

    DDT: A Diffusion-Driven Transformer-based Framework for Human Mesh Recovery from a Video. (arXiv:2303.13397v1 [cs.CV])

    [http://arxiv.org/abs/2303.13397](http://arxiv.org/abs/2303.13397)

    提出了一种基于扩散驱动变压器的视频 HMR 框架（DDT），它旨在从输入序列中解码特定的运动模式，增强运动平滑性和时间一致性，并输出所有帧的人体网格，使得 DDT 更适用于时间效率至关重要的实际应用。

    

    人体网格恢复（HMR）为各种实际应用提供了丰富的人体信息，例如游戏、人机交互和虚拟现实。与单一图像方法相比，基于视频的方法可以利用时间信息通过融合人体运动先验进一步提高性能。然而，像 VIBE 这样的多对多方法存在运动平滑性和时间一致性的挑战。而像 TCMR 和 MPS-Net 这样的多对一方法则依赖于未来帧，在推理过程中是非因果和时间效率低下的。为了解决这些挑战，提出了一种新的基于扩散驱动变压器的视频 HMR 框架（DDT）。DDT 旨在从输入序列中解码特定的运动模式，增强运动平滑性和时间一致性。作为一种多对多方法，DDT 的解码器输出所有帧的人体网格，使 DDT 更适用于时间效率至关重要的实际应用。

    Human mesh recovery (HMR) provides rich human body information for various real-world applications such as gaming, human-computer interaction, and virtual reality. Compared to single image-based methods, video-based methods can utilize temporal information to further improve performance by incorporating human body motion priors. However, many-to-many approaches such as VIBE suffer from motion smoothness and temporal inconsistency. While many-to-one approaches such as TCMR and MPS-Net rely on the future frames, which is non-causal and time inefficient during inference. To address these challenges, a novel Diffusion-Driven Transformer-based framework (DDT) for video-based HMR is presented. DDT is designed to decode specific motion patterns from the input sequence, enhancing motion smoothness and temporal consistency. As a many-to-many approach, the decoder of our DDT outputs the human mesh of all the frames, making DDT more viable for real-world applications where time efficiency is cruc
    

