# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GET: Unlocking the Multi-modal Potential of CLIP for Generalized Category Discovery](https://arxiv.org/abs/2403.09974) | 本文提出了一种文本嵌入合成器（TES），用于为无标签数据生成伪文本嵌入，以解锁CLIP用于广义类别发现任务中的多模态潜力。 |
| [^2] | [Review of the Learning-based Camera and Lidar Simulation Methods for Autonomous Driving Systems](https://arxiv.org/abs/2402.10079) | 本文综述了自主驾驶系统中基于学习的相机和激光雷达仿真方法的最新研究现状。 |
| [^3] | [SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models](https://arxiv.org/abs/2402.05935) | 本论文介绍了SPHINX-X，一种扩展的多模态大型语言模型系列。通过改进架构和训练效率，我们成功构建了一系列参数大小和多语言能力不同的MLLMs，与数据和参数规模有强相关性。 |
| [^4] | [AutArch: An AI-assisted workflow for object detection and automated recording in archaeological catalogues](https://arxiv.org/abs/2311.17978) | 这篇论文介绍了AutArch，一种用于考古目录中物体检测和自动化记录的人工智能辅助工作流程，并提出了一种新的数据收集方法，通过自动化从遗留资源中提取数据，解决了现有记录质量和标准不一致的挑战。 |
| [^5] | [Karyotype AI for Precision Oncology.](http://arxiv.org/abs/2211.14312) | 本研究针对精准肿瘤学中的染色体分析问题，通过使用Fred Hutchinson癌症研究中心的大量数据，利用深度学习模型和拓扑视觉转换器(TopViTs)，成功开发出了一种自动识别染色体异常的方法。 |

# 详细

[^1]: GET：解锁CLIP的多模态潜力，用于广义类别发现

    GET: Unlocking the Multi-modal Potential of CLIP for Generalized Category Discovery

    [https://arxiv.org/abs/2403.09974](https://arxiv.org/abs/2403.09974)

    本文提出了一种文本嵌入合成器（TES），用于为无标签数据生成伪文本嵌入，以解锁CLIP用于广义类别发现任务中的多模态潜力。

    

    给定包含旧类别和新类别的无标签数据集，广义类别发现（GCD）旨在准确发现新类别，并正确分类旧类别，利用从有标签样本中学习的类别概念。当前的GCD方法只使用单一的视觉信息模态，导致在视觉上相似类别的分类效果不佳。虽然某些类别在视觉上容易混淆，但它们的文本信息可能是不同的，这促使我们将文本信息引入到GCD任务中。然而，无标签数据缺乏类别名称，使得利用文本信息变得不切实际。为了解决这一具有挑战性的问题，在本文中，我们提出了一种文本嵌入合成器（TES），用于为无标签样本生成伪文本嵌入。具体而言，我们的TES利用CLIP可以生成对齐的视觉-语言特征这一特性，将视觉嵌入转换为CLIP文本模型的标记。

    arXiv:2403.09974v1 Announce Type: cross  Abstract: Given unlabelled datasets containing both old and new categories, generalized category discovery (GCD) aims to accurately discover new classes while correctly classifying old classes, leveraging the class concepts learned from labeled samples. Current GCD methods only use a single visual modality of information, resulting in poor classification of visually similar classes. Though certain classes are visually confused, their text information might be distinct, motivating us to introduce text information into the GCD task. However, the lack of class names for unlabelled data makes it impractical to utilize text information. To tackle this challenging problem, in this paper, we propose a Text Embedding Synthesizer (TES) to generate pseudo text embeddings for unlabelled samples. Specifically, our TES leverages the property that CLIP can generate aligned vision-language features, converting visual embeddings into tokens of the CLIP's text e
    
[^2]: 自主驾驶系统中基于学习的相机和激光雷达仿真方法的综述

    Review of the Learning-based Camera and Lidar Simulation Methods for Autonomous Driving Systems

    [https://arxiv.org/abs/2402.10079](https://arxiv.org/abs/2402.10079)

    本文综述了自主驾驶系统中基于学习的相机和激光雷达仿真方法的最新研究现状。

    

    感知传感器，尤其是相机和激光雷达，是自主驾驶系统(Autonomous Driving Systems，ADS)的关键元素，使其能够理解周围环境以做出明智的驾驶和控制决策。因此，开发逼真的相机和激光雷达模拟方法，也称为相机和激光雷达模型，对于有效进行基于仿真的ADS测试至关重要。此外，基于深度学习的感知模型的兴起，促进了感知传感器模型作为合成各种训练数据集的有价值工具的普及。传统传感器仿真方法依赖于计算密集型的基于物理的算法，特别是在复杂系统如ADS中。因此，目前的潜力在于基于学习的模型，受到深度生成模型在合成高维数据方面取得成功的推动。本文综述了基于学习的传感器仿真方法的最新研究现状。

    arXiv:2402.10079v1 Announce Type: cross  Abstract: Perception sensors, particularly camera and Lidar, are key elements of Autonomous Driving Systems (ADS) that enable them to comprehend their surroundings for informed driving and control decisions. Therefore, developing realistic camera and Lidar simulation methods, also known as camera and Lidar models, is of paramount importance to effectively conduct simulation-based testing for ADS. Moreover, the rise of deep learning-based perception models has propelled the prevalence of perception sensor models as valuable tools for synthesising diverse training datasets. The traditional sensor simulation methods rely on computationally expensive physics-based algorithms, specifically in complex systems such as ADS. Hence, the current potential resides in learning-based models, driven by the success of deep generative models in synthesising high-dimensional data. This paper reviews the current state-of-the-art in learning-based sensor simulation
    
[^3]: SPHINX-X: 扩展数据和参数用于一系列多模态大型语言模型

    SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models

    [https://arxiv.org/abs/2402.05935](https://arxiv.org/abs/2402.05935)

    本论文介绍了SPHINX-X，一种扩展的多模态大型语言模型系列。通过改进架构和训练效率，我们成功构建了一系列参数大小和多语言能力不同的MLLMs，与数据和参数规模有强相关性。

    

    我们提出SPHINX-X，一种基于SPHINX开发的广泛多模态大型语言模型（MLLM）系列。为了改善架构和训练效率，我们通过移除冗余的视觉编码器、绕过完全填充的子图像，并将多阶段训练简化成为一阶段的全集合模式，修改了SPHINX框架。为了充分发挥MLLM的潜力，我们组装了一个综合的跨语言、跨视觉和视觉-语言任务的多领域、多模态的数据集，涵盖了公开可用的资源。我们进一步使用我们的OCR密集和Mark数据集丰富这个收集，扩展了多样性和普适性。通过对不同基础LLM进行训练，包括TinyLlama1.1B、InternLM2-7B、LLaMA2-13B和Mixtral8x7B，我们获得了一系列参数大小和多语言能力变化的MLLMs。全面的基准测试揭示了多模态性能与数据和参数规模之间的强相关性。

    We propose SPHINX-X, an extensive Multimodality Large Language Model (MLLM) series developed upon SPHINX. To improve the architecture and training efficiency, we modify the SPHINX framework by removing redundant visual encoders, bypassing fully-padded sub-images with skip tokens, and simplifying multi-stage training into a one-stage all-in-one paradigm. To fully unleash the potential of MLLMs, we assemble a comprehensive multi-domain and multimodal dataset covering publicly available resources in language, vision, and vision-language tasks. We further enrich this collection with our curated OCR intensive and Set-of-Mark datasets, extending the diversity and generality. By training over different base LLMs including TinyLlama1.1B, InternLM2-7B, LLaMA2-13B, and Mixtral8x7B, we obtain a spectrum of MLLMs that vary in parameter size and multilingual capabilities. Comprehensive benchmarking reveals a strong correlation between the multi-modal performance with the data and parameter scales. 
    
[^4]: AutArch：一种用于考古目录中物体检测和自动化记录的人工智能辅助工作流程

    AutArch: An AI-assisted workflow for object detection and automated recording in archaeological catalogues

    [https://arxiv.org/abs/2311.17978](https://arxiv.org/abs/2311.17978)

    这篇论文介绍了AutArch，一种用于考古目录中物体检测和自动化记录的人工智能辅助工作流程，并提出了一种新的数据收集方法，通过自动化从遗留资源中提取数据，解决了现有记录质量和标准不一致的挑战。

    

    这篇论文的背景是利用人工智能和大数据从异构的已发表资源中创建大规模统一的考古数据集，比如遗物目录。论文关注的是一致考古数据组合的挑战。由于现有记录在质量和记录标准上存在差异，我们无法简单地合并现有记录。因此，必须从已发表的考古插图中重新创建记录。只有通过自动化的帮助，这才是可行的途径。本文的贡献是一个新的工作流程，用于从考古遗物目录中收集数据，这些目录作为遗留资源存在，比如大型未排序的PDF文件中的考古绘图和照片；该工作流程依赖于支持图像处理、物体检测以及验证和调整自动获取数据的交互手段的自定义软件（AutArch）。我们集成了人工智能技术。

    arXiv:2311.17978v2 Announce Type: replace-cross  Abstract: The context of this paper is the creation of large uniform archaeological datasets from heterogeneous published resources, such as find catalogues - with the help of AI and Big Data. The paper is concerned with the challenge of consistent assemblages of archaeological data. We cannot simply combine existing records, as they differ in terms of quality and recording standards. Thus, records have to be recreated from published archaeological illustrations. This is only a viable path with the help of automation. The contribution of this paper is a new workflow for collecting data from archaeological find catalogues available as legacy resources, such as archaeological drawings and photographs in large unsorted PDF files; the workflow relies on custom software (AutArch) supporting image processing, object detection, and interactive means of validating and adjusting automatically retrieved data. We integrate artificial intelligence (
    
[^5]: 精准肿瘤学的染色体AI

    Karyotype AI for Precision Oncology. (arXiv:2211.14312v3 [q-bio.QM] UPDATED)

    [http://arxiv.org/abs/2211.14312](http://arxiv.org/abs/2211.14312)

    本研究针对精准肿瘤学中的染色体分析问题，通过使用Fred Hutchinson癌症研究中心的大量数据，利用深度学习模型和拓扑视觉转换器(TopViTs)，成功开发出了一种自动识别染色体异常的方法。

    

    染色体分析对于诊断遗传疾病至关重要。对于血液系统恶性肿瘤，通过染色体组型分析来发现体细胞突变是标准的护理方法。然而，染色体组型分析因为大部分是手动操作，且需要专业知识来识别和注释突变，所以昂贵且耗时。以Fred Hutchinson癌症研究中心过去五年的约10,000个患者标本和约50,000个染色体组型图片作为训练集，我们创建了一组代表单个染色体的标记图片。这些单个染色体用于训练和评估深度学习模型，以分类人类的24条染色体和识别染色体异常。具有最高准确性的模型使用了最近引入的拓扑视觉转换器(TopViTs)和二级块-托普利茨蒙版，以融入结构性归纳偏置。TopViT的性能优于CNN(Inc)

    Chromosome analysis is essential for diagnosing genetic disorders. For hematologic malignancies, identification of somatic clonal aberrations by karyotype analysis remains the standard of care. However, karyotyping is costly and time-consuming because of the largely manual process and the expertise required in identifying and annotating aberrations. Efforts to automate karyotype analysis to date fell short in aberration detection. Using a training set of ~10k patient specimens and ~50k karyograms from over 5 years from the Fred Hutchinson Cancer Center, we created a labeled set of images representing individual chromosomes. These individual chromosomes were used to train and assess deep learning models for classifying the 24 human chromosomes and identifying chromosomal aberrations. The top-accuracy models utilized the recently introduced Topological Vision Transformers (TopViTs) with 2-level-block-Toeplitz masking, to incorporate structural inductive bias. TopViT outperformed CNN (Inc
    

