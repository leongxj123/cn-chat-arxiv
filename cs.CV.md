# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation](https://arxiv.org/abs/2403.19103) | PRISM是一种算法，可以自动识别人类可解释且易传递的提示，从而有效生成所需概念，仅使用黑盒访问T2I模型。 |
| [^2] | [Quality-Aware Image-Text Alignment for Real-World Image Quality Assessment](https://arxiv.org/abs/2403.11176) | 提出了一种基于CLIP的自监督方法QualiCLIP，通过质量感知的图像-文本对齐策略，实现了图像质量评估不需要标记MOS的问题 |
| [^3] | [RealDex: Towards Human-like Grasping for Robotic Dexterous Hand](https://arxiv.org/abs/2402.13853) | RealDex数据集捕捉了真实的灵巧手抓取动作，利用多模态数据使得训练灵巧手更加自然和精确，同时提出了一种先进的灵巧抓取动作生成框架，有效利用多模态大型语言模型，在类人机器人的自动感知、认知和操纵方面具有巨大潜力。 |
| [^4] | [InkSight: Offline-to-Online Handwriting Conversion by Learning to Read and Write](https://arxiv.org/abs/2402.05804) | InkSight是一个可以将离线手写转换为在线手写的系统，通过结合阅读和书写先验知识，在多样化的照片中有效地Derendering手写文本。 |
| [^5] | [CIC: A framework for Culturally-aware Image Captioning](https://arxiv.org/abs/2402.05374) | CIC是一种面向文化感知图像字幕的框架，通过结合视觉问答和大型语言模型，它能够生成能描述图像中文化元素的详细字幕。 |
| [^6] | [ActiveAnno3D - An Active Learning Framework for Multi-Modal 3D Object Detection](https://arxiv.org/abs/2402.03235) | 这项工作提出了一种用于多模态3D物体检测的主动学习框架ActiveAnno3D。通过选择最具信息量的训练数据样本进行标注，我们能够在使用一半的训练数据时实现与传统方法相近的检测性能。 |
| [^7] | [GD-CAF: Graph Dual-stream Convolutional Attention Fusion for Precipitation Nowcasting](https://arxiv.org/abs/2401.07958) | GD-CAF提出了一种新颖的方法，将降水预报作为一个时空图序列预报问题，利用图形双流卷积注意力融合来学习历史降水图并在不同空间位置上预测未来的降水。 |
| [^8] | [GeoSAM: Fine-tuning SAM with Sparse and Dense Visual Prompting for Automated Segmentation of Mobility Infrastructure](https://arxiv.org/abs/2311.11319) | GeoSAM是一个基于SAM的新框架，使用了来自零样本学习和预训练CNN分割模型的视觉提示，提高了地理图像分割的性能。 |
| [^9] | [Domain Generalization for Medical Image Analysis: A Survey.](http://arxiv.org/abs/2310.08598) | 本综述详细回顾了针对医学图像分析的领域泛化研究，探讨了在DL模型在真实世界应用中遇到的挑战，以及如何解决分布漂移问题和实现稳健性。同时，考虑了领域泛化技术对整个MedIA工作流程的操作影响。 |
| [^10] | [Comprehensive Assessment of the Performance of Deep Learning Classifiers Reveals a Surprising Lack of Robustness.](http://arxiv.org/abs/2308.04137) | 通过综合评估深度学习分类器的性能，发现它们缺乏稳定性和可靠性，并建议采用广泛的数据类型和统一的评估指标进行性能基准测试。 |
| [^11] | [Your Room is not Private: Gradient Inversion Attack on Reinforcement Learning.](http://arxiv.org/abs/2306.09273) | 这篇论文提出了一种针对值函数算法和梯度算法的攻击方法，利用梯度反转重建状态、动作和监督信号，以解决嵌入式人工智能中的隐私泄露问题。 |
| [^12] | [Leveraging the Triple Exponential Moving Average for Fast-Adaptive Moment Estimation.](http://arxiv.org/abs/2306.01423) | 本文提出了一种新的深度优化器FAME，使用三重指数移动平均值（TEMA）来估计梯度矩，提供更丰富和准确的数据变化和趋势信息，可以提高计算机视觉等领域中模型的性能表现。 |

# 详细

[^1]: 用于个性化文本到图像生成的自动化黑盒提示工程

    Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation

    [https://arxiv.org/abs/2403.19103](https://arxiv.org/abs/2403.19103)

    PRISM是一种算法，可以自动识别人类可解释且易传递的提示，从而有效生成所需概念，仅使用黑盒访问T2I模型。

    

    提示工程对于控制文本到图像（T2I）生成模型的输出是有效的，但由于需要手动制作提示而导致工作繁重。这一挑战促使了自动提示生成算法的发展。然而，这些方法通常在T2I模型之间的可传递性方面遇到困难，需要对基础模型进行白盒访问，并产生非直观的提示。在这项工作中，我们介绍了PRISM，这是一种算法，可以仅使用黑盒访问T2I模型就自动识别人类可解释且易传递的提示，从而有效生成所需概念。受大型语言模型（LLM）越狱的启发，PRISM利用LLM的上下文学习能力来迭代地改进给定参考图像的候选提示分布。我们的实验展示了PRISM在为对象、样式等生成准确提示方面的多样性和有效性。

    arXiv:2403.19103v1 Announce Type: cross  Abstract: Prompt engineering is effective for controlling the output of text-to-image (T2I) generative models, but it is also laborious due to the need for manually crafted prompts. This challenge has spurred the development of algorithms for automated prompt generation. However, these methods often struggle with transferability across T2I models, require white-box access to the underlying model, and produce non-intuitive prompts. In this work, we introduce PRISM, an algorithm that automatically identifies human-interpretable and transferable prompts that can effectively generate desired concepts given only black-box access to T2I models. Inspired by large language model (LLM) jailbreaking, PRISM leverages the in-context learning ability of LLMs to iteratively refine the candidate prompts distribution for given reference images. Our experiments demonstrate the versatility and effectiveness of PRISM in generating accurate prompts for objects, sty
    
[^2]: 面向现实世界图像质量评估的质量感知图像-文本对齐

    Quality-Aware Image-Text Alignment for Real-World Image Quality Assessment

    [https://arxiv.org/abs/2403.11176](https://arxiv.org/abs/2403.11176)

    提出了一种基于CLIP的自监督方法QualiCLIP，通过质量感知的图像-文本对齐策略，实现了图像质量评估不需要标记MOS的问题

    

    无参考图像质量评估（NR-IQA）致力于设计一种在没有高质量参考图像的情况下测量图像质量的方法，以符合人类感知，大部分最先进的NR-IQA方法中依赖标注的主观评分（MOS）限制了它们在真实场景中的可扩展性和广泛适用性。为了克服这一限制，我们提出了QualiCLIP（Quality-aware CLIP），这是一种基于CLIP的自监督不需要标记MOS的方法。具体来说，我们引入了一种质量感知的图像-文本对齐策略，使得CLIP生成的表示与图像固有质量相关。从原始图像开始，我们使用不断增加的强度合成地劣化它们。然后，我们训练CLIP根据其与质量相关的反义文本提示的相似性对这些降解图像进行排名，同时保证一致的表达

    arXiv:2403.11176v1 Announce Type: cross  Abstract: No-Reference Image Quality Assessment (NR-IQA) focuses on designing methods to measure image quality in alignment with human perception when a high-quality reference image is unavailable. The reliance on annotated Mean Opinion Scores (MOS) in the majority of state-of-the-art NR-IQA approaches limits their scalability and broader applicability to real-world scenarios. To overcome this limitation, we propose QualiCLIP (Quality-aware CLIP), a CLIP-based self-supervised opinion-unaware method that does not require labeled MOS. In particular, we introduce a quality-aware image-text alignment strategy to make CLIP generate representations that correlate with the inherent quality of the images. Starting from pristine images, we synthetically degrade them with increasing levels of intensity. Then, we train CLIP to rank these degraded images based on their similarity to quality-related antonym text prompts, while guaranteeing consistent represe
    
[^3]: RealDex: 实现机器人灵巧手类人式抓取

    RealDex: Towards Human-like Grasping for Robotic Dexterous Hand

    [https://arxiv.org/abs/2402.13853](https://arxiv.org/abs/2402.13853)

    RealDex数据集捕捉了真实的灵巧手抓取动作，利用多模态数据使得训练灵巧手更加自然和精确，同时提出了一种先进的灵巧抓取动作生成框架，有效利用多模态大型语言模型，在类人机器人的自动感知、认知和操纵方面具有巨大潜力。

    

    在本文中，我们介绍了RealDex，一个开创性的数据集，捕捉了融入了人类行为模式的真实灵巧手抓取动作，同时通过多视角和多模态视觉数据进行了丰富。利用远程操作系统，我们可以实时无缝同步人-机器人手姿势。这些类人动作的集合对于训练灵巧手更自然、更精确地模仿人类动作至关重要。RealDex在推动类人机器人在真实场景中自动感知、认知和操纵方面具有巨大潜力。此外，我们介绍了一种前沿的灵巧抓取动作生成框架，该框架符合人类经验，并通过有效利用多模态大型语言模型增强了在现实世界中的适用性。广泛的实验证明了我们的方法在RealDex和其他开放数据集上的优越性能。完整的数据集和代码将会公开发布。

    arXiv:2402.13853v1 Announce Type: cross  Abstract: In this paper, we introduce RealDex, a pioneering dataset capturing authentic dexterous hand grasping motions infused with human behavioral patterns, enriched by multi-view and multimodal visual data. Utilizing a teleoperation system, we seamlessly synchronize human-robot hand poses in real time. This collection of human-like motions is crucial for training dexterous hands to mimic human movements more naturally and precisely. RealDex holds immense promise in advancing humanoid robot for automated perception, cognition, and manipulation in real-world scenarios. Moreover, we introduce a cutting-edge dexterous grasping motion generation framework, which aligns with human experience and enhances real-world applicability through effectively utilizing Multimodal Large Language Models. Extensive experiments have demonstrated the superior performance of our method on RealDex and other open datasets. The complete dataset and code will be made 
    
[^4]: InkSight：通过学习阅读和书写实现离线到在线手写转换

    InkSight: Offline-to-Online Handwriting Conversion by Learning to Read and Write

    [https://arxiv.org/abs/2402.05804](https://arxiv.org/abs/2402.05804)

    InkSight是一个可以将离线手写转换为在线手写的系统，通过结合阅读和书写先验知识，在多样化的照片中有效地Derendering手写文本。

    

    数字笔记正在变得越来越受欢迎，提供了一种耐用、可编辑和易于索引的存储笔记的方式，即矢量化形式的数字墨水。然而，这种笔记方式与传统的纸笔记方式之间仍存在显著差距，而传统纸笔记方式仍受到绝大多数人的青睐。我们的工作InkSight旨在弥合这种差距，使实体笔记者能够轻松地将他们的作品（离线手写）转换为数字墨水（在线手写），这个过程我们称之为Derendering。之前关于此主题的研究集中在图像的几何属性上，导致了在训练领域之外的有限泛化能力。我们的方法结合了阅读和书写的先验知识，允许在缺乏大量配对样本的情况下训练模型，而这些配对样本很难获取。据我们所知，这是第一个有效地对具有多样化视觉特征和背景的任意照片中的手写文本进行Derendering的工作。

    Digital note-taking is gaining popularity, offering a durable, editable, and easily indexable way of storing notes in the vectorized form, known as digital ink. However, a substantial gap remains between this way of note-taking and traditional pen-and-paper note-taking, a practice still favored by a vast majority. Our work, InkSight, aims to bridge the gap by empowering physical note-takers to effortlessly convert their work (offline handwriting) to digital ink (online handwriting), a process we refer to as Derendering. Prior research on the topic has focused on the geometric properties of images, resulting in limited generalization beyond their training domains. Our approach combines reading and writing priors, allowing training a model in the absence of large amounts of paired samples, which are difficult to obtain. To our knowledge, this is the first work that effectively derenders handwritten text in arbitrary photos with diverse visual characteristics and backgrounds. Furthermore,
    
[^5]: CIC：一种面向文化感知图像字幕的框架

    CIC: A framework for Culturally-aware Image Captioning

    [https://arxiv.org/abs/2402.05374](https://arxiv.org/abs/2402.05374)

    CIC是一种面向文化感知图像字幕的框架，通过结合视觉问答和大型语言模型，它能够生成能描述图像中文化元素的详细字幕。

    

    图像字幕通过使用视觉-语言预训练模型（VLPs）如BLIP从图像生成描述性句子，这种方法已经取得了很大的改进。然而，当前的方法缺乏对图像中所描绘的文化元素（例如亚洲文化群体的传统服装）生成详细描述性字幕的能力。在本文中，我们提出了一种新的框架，\textbf{面向文化感知图像字幕（CIC）}，该框架能够从代表不同文化的图像中生成字幕并描述文化元素。受到将视觉模态和大型语言模型（LLMs）通过适当的提示进行组合的方法的启发，我们的框架（1）根据图像中的文化类别生成问题，（2）利用生成的问题从视觉问答（VQA）中提取文化视觉元素，（3）使用带有提示的LLMs生成文化感知字幕。我们在4个不同大学的45名参与者上进行了人工评估。

    Image Captioning generates descriptive sentences from images using Vision-Language Pre-trained models (VLPs) such as BLIP, which has improved greatly. However, current methods lack the generation of detailed descriptive captions for the cultural elements depicted in the images, such as the traditional clothing worn by people from Asian cultural groups. In this paper, we propose a new framework, \textbf{Culturally-aware Image Captioning (CIC)}, that generates captions and describes cultural elements extracted from cultural visual elements in images representing cultures. Inspired by methods combining visual modality and Large Language Models (LLMs) through appropriate prompts, our framework (1) generates questions based on cultural categories from images, (2) extracts cultural visual elements from Visual Question Answering (VQA) using generated questions, and (3) generates culturally-aware captions using LLMs with the prompts. Our human evaluation conducted on 45 participants from 4 dif
    
[^6]: ActiveAnno3D - 一种用于多模态3D物体检测的主动学习框架

    ActiveAnno3D - An Active Learning Framework for Multi-Modal 3D Object Detection

    [https://arxiv.org/abs/2402.03235](https://arxiv.org/abs/2402.03235)

    这项工作提出了一种用于多模态3D物体检测的主动学习框架ActiveAnno3D。通过选择最具信息量的训练数据样本进行标注，我们能够在使用一半的训练数据时实现与传统方法相近的检测性能。

    

    大规模数据集的筛选仍然需要大量的时间和资源，数据通常需要人工标注，创建高质量数据集的难题依然存在。在这项工作中，我们使用主动学习的方法来解决多模态3D物体检测中的研究空白。我们提出了ActiveAnno3D，一个用于选择最具信息量的训练数据样本进行标注的主动学习框架。我们探索了各种连续训练方法，并集成了在计算需求和检测性能方面最高效的方法。此外，我们对nuScenes和TUM Traffic Intersection数据集进行了大量实验和消融研究，使用BEVFusion和PV-RCNN进行了测试。我们展示了当仅使用TUM Traffic Intersection数据集的一半训练数据（77.25 mAP相比于83.50 mAP）时，使用PV-RCNN和基于熵的查询策略几乎可以达到相同的性能，而BEVFusion则在使用一半的训练数据时获得了64.31的mAP。

    The curation of large-scale datasets is still costly and requires much time and resources. Data is often manually labeled, and the challenge of creating high-quality datasets remains. In this work, we fill the research gap using active learning for multi-modal 3D object detection. We propose ActiveAnno3D, an active learning framework to select data samples for labeling that are of maximum informativeness for training. We explore various continuous training methods and integrate the most efficient method regarding computational demand and detection performance. Furthermore, we perform extensive experiments and ablation studies with BEVFusion and PV-RCNN on the nuScenes and TUM Traffic Intersection dataset. We show that we can achieve almost the same performance with PV-RCNN and the entropy-based query strategy when using only half of the training data (77.25 mAP compared to 83.50 mAP) of the TUM Traffic Intersection dataset. BEVFusion achieved an mAP of 64.31 when using half of the trai
    
[^7]: GD-CAF：用于降水预报的图形双流卷积注意力融合

    GD-CAF: Graph Dual-stream Convolutional Attention Fusion for Precipitation Nowcasting

    [https://arxiv.org/abs/2401.07958](https://arxiv.org/abs/2401.07958)

    GD-CAF提出了一种新颖的方法，将降水预报作为一个时空图序列预报问题，利用图形双流卷积注意力融合来学习历史降水图并在不同空间位置上预测未来的降水。

    

    精确的降水预报对于各种应用至关重要，包括洪水预测、灾害管理、优化农业活动、管理交通路线和可再生能源。本文将降水预报形式化为时空图序列预报问题，提出了一种名为图形双流卷积注意力融合（GD-CAF）的新方法，旨在从历史降水图的时空图中学习，并预测未来不同空间位置的降水。

    arXiv:2401.07958v2 Announce Type: replace  Abstract: Accurate precipitation nowcasting is essential for various applications, including flood prediction, disaster management, optimizing agricultural activities, managing transportation routes and renewable energy. While several studies have addressed this challenging task from a sequence-to-sequence perspective, most of them have focused on a single area without considering the existing correlation between multiple disjoint regions. In this paper, we formulate precipitation nowcasting as a spatiotemporal graph sequence nowcasting problem. In particular, we introduce Graph Dual-stream Convolutional Attention Fusion (GD-CAF), a novel approach designed to learn from historical spatiotemporal graph of precipitation maps and nowcast future time step ahead precipitation at different spatial locations. GD-CAF consists of spatio-temporal convolutional attention as well as gated fusion modules which are equipped with depthwise-separable convolut
    
[^8]: GeoSAM: 使用稀疏和密集的视觉提示对SAM进行改进，实现自动化的移动基础设施分割

    GeoSAM: Fine-tuning SAM with Sparse and Dense Visual Prompting for Automated Segmentation of Mobility Infrastructure

    [https://arxiv.org/abs/2311.11319](https://arxiv.org/abs/2311.11319)

    GeoSAM是一个基于SAM的新框架，使用了来自零样本学习和预训练CNN分割模型的视觉提示，提高了地理图像分割的性能。

    

    当应用于自然图像分割时，Segment Anything Model (SAM)已经展现出了令人印象深刻的性能。然而，它在地理图像（如航拍和卫星图像）中面临困难，特别是在分割道路、人行道和人行横道等移动基础设施时。这种较差的性能源于这些对象的窄小特征，它们的纹理融入环境中，以及树木、建筑物、车辆和行人等物体的干扰，这些都可能使模型失去定向产生不准确的分割图。为了解决这些挑战，我们提出了地理SAM（GeoSAM），这是一个基于SAM的新框架，它使用来自零样本学习的密集视觉提示和预训练CNN分割模型的稀疏视觉提示实施了细调策略。所提出的GeoSAM在地理图像分割方面优于现有方法，特别是对于道路基础设施、行人基础设施的分割性能提升了26％、7％和17％。

    The Segment Anything Model (SAM) has shown impressive performance when applied to natural image segmentation. However, it struggles with geographical images like aerial and satellite imagery, especially when segmenting mobility infrastructure including roads, sidewalks, and crosswalks. This inferior performance stems from the narrow features of these objects, their textures blending into the surroundings, and interference from objects like trees, buildings, vehicles, and pedestrians - all of which can disorient the model to produce inaccurate segmentation maps. To address these challenges, we propose Geographical SAM (GeoSAM), a novel SAM-based framework that implements a fine-tuning strategy using the dense visual prompt from zero-shot learning, and the sparse visual prompt from a pre-trained CNN segmentation model. The proposed GeoSAM outperforms existing approaches for geographical image segmentation, specifically by 26%, 7%, and 17% for road infrastructure, pedestrian infrastructur
    
[^9]: 医学图像分析的领域泛化：综述

    Domain Generalization for Medical Image Analysis: A Survey. (arXiv:2310.08598v1 [eess.IV])

    [http://arxiv.org/abs/2310.08598](http://arxiv.org/abs/2310.08598)

    本综述详细回顾了针对医学图像分析的领域泛化研究，探讨了在DL模型在真实世界应用中遇到的挑战，以及如何解决分布漂移问题和实现稳健性。同时，考虑了领域泛化技术对整个MedIA工作流程的操作影响。

    

    医学图像分析（MedIA）已成为医学和保健领域的重要工具，在疾病诊断、预后和治疗规划方面起到了很大的作用，深度学习（DL）的最新成功为其进展做出了重要贡献。然而，MedIA的DL模型在现实世界中的部署仍然具有挑战性，在训练和测试样本之间的分布差异下很难泛化，这被称为分布漂移问题。研究人员致力于开发各种DL方法，使其能够适应并在未知和超出分布的数据分布上稳健地运行。本文综合评述了专门针对MedIA的领域泛化研究。我们提供了领域泛化技术在更大范围MedIA系统内的交互方式的整体视图，不仅仅考虑方法学，还考虑了对整个MedIA工作流程的操作影响。具体而言，我们将领域泛化方法分为数据层次的方法…

    Medical Image Analysis (MedIA) has become an essential tool in medicine and healthcare, aiding in disease diagnosis, prognosis, and treatment planning, and recent successes in deep learning (DL) have made significant contributions to its advances. However, DL models for MedIA remain challenging to deploy in real-world situations, failing for generalization under the distributional gap between training and testing samples, known as a distribution shift problem. Researchers have dedicated their efforts to developing various DL methods to adapt and perform robustly on unknown and out-of-distribution data distributions. This paper comprehensively reviews domain generalization studies specifically tailored for MedIA. We provide a holistic view of how domain generalization techniques interact within the broader MedIA system, going beyond methodologies to consider the operational implications on the entire MedIA workflow. Specifically, we categorize domain generalization methods into data-lev
    
[^10]: 深度学习分类器性能的综合评估揭示出惊人的缺乏稳定性

    Comprehensive Assessment of the Performance of Deep Learning Classifiers Reveals a Surprising Lack of Robustness. (arXiv:2308.04137v1 [cs.LG])

    [http://arxiv.org/abs/2308.04137](http://arxiv.org/abs/2308.04137)

    通过综合评估深度学习分类器的性能，发现它们缺乏稳定性和可靠性，并建议采用广泛的数据类型和统一的评估指标进行性能基准测试。

    

    可靠而稳健的评估方法是开发本身稳健可靠的机器学习模型的必要第一步。然而，目前用于评估分类器的常规评估协议在综合评估性能方面存在不足，因为它们往往依赖于有限类型的测试数据，忽视其他类型的数据。例如，使用标准测试数据无法评估分类器对于未经训练的类别样本的预测。另一方面，使用包含未知类别样本的数据进行测试无法评估分类器对于已知类别标签的预测能力。本文提倡使用各种不同类型的数据进行性能基准测试，并使用一种可应用于所有这些数据类型的单一指标，以产生一致的性能评估结果。通过这样的基准测试发现，目前的深度神经网络，包括使用认为是全面的方法进行训练的网络，也存在缺乏稳定性的问题。

    Reliable and robust evaluation methods are a necessary first step towards developing machine learning models that are themselves robust and reliable. Unfortunately, current evaluation protocols typically used to assess classifiers fail to comprehensively evaluate performance as they tend to rely on limited types of test data, and ignore others. For example, using the standard test data fails to evaluate the predictions made by the classifier to samples from classes it was not trained on. On the other hand, testing with data containing samples from unknown classes fails to evaluate how well the classifier can predict the labels for known classes. This article advocates bench-marking performance using a wide range of different types of data and using a single metric that can be applied to all such data types to produce a consistent evaluation of performance. Using such a benchmark it is found that current deep neural networks, including those trained with methods that are believed to pro
    
[^11]: 你的房间不是私密的：关于强化学习的梯度反转攻击

    Your Room is not Private: Gradient Inversion Attack on Reinforcement Learning. (arXiv:2306.09273v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2306.09273](http://arxiv.org/abs/2306.09273)

    这篇论文提出了一种针对值函数算法和梯度算法的攻击方法，利用梯度反转重建状态、动作和监督信号，以解决嵌入式人工智能中的隐私泄露问题。

    

    嵌入式人工智能的显著发展吸引了人们的极大关注，该技术使得机器人可以在虚拟环境中导航、感知和互动。由于计算机视觉和大型语言模型方面的显著进展，隐私问题在嵌入式人工智能领域变得至关重要，因为机器人可以访问大量个人信息。然而，关于强化学习算法中的隐私泄露问题，尤其是关于值函数算法和梯度算法的问题，在研究中尚未得到充分考虑。本文旨在通过提出一种攻击值函数算法和梯度算法的方法，利用梯度反转重建状态、动作和监督信号，来解决这一问题。选择使用梯度进行攻击是因为常用的联邦学习技术仅利用基于私人用户数据计算的梯度来优化模型，而不存储或传输用户数据。

    The prominence of embodied Artificial Intelligence (AI), which empowers robots to navigate, perceive, and engage within virtual environments, has attracted significant attention, owing to the remarkable advancements in computer vision and large language models. Privacy emerges as a pivotal concern within the realm of embodied AI, as the robot accesses substantial personal information. However, the issue of privacy leakage in embodied AI tasks, particularly in relation to reinforcement learning algorithms, has not received adequate consideration in research. This paper aims to address this gap by proposing an attack on the value-based algorithm and the gradient-based algorithm, utilizing gradient inversion to reconstruct states, actions, and supervision signals. The choice of using gradients for the attack is motivated by the fact that commonly employed federated learning techniques solely utilize gradients computed based on private user data to optimize models, without storing or trans
    
[^12]: 利用三重指数移动平均值实现快速自适应矩估计

    Leveraging the Triple Exponential Moving Average for Fast-Adaptive Moment Estimation. (arXiv:2306.01423v1 [cs.CV])

    [http://arxiv.org/abs/2306.01423](http://arxiv.org/abs/2306.01423)

    本文提出了一种新的深度优化器FAME，使用三重指数移动平均值（TEMA）来估计梯度矩，提供更丰富和准确的数据变化和趋势信息，可以提高计算机视觉等领域中模型的性能表现。

    

    网络优化是深度学习领域中的一个关键步骤，直接影响计算机视觉等多种领域中模型的性能。虽然多种优化器已经被开发出来，但目前的方法在准确快速地识别梯度趋势方面仍然有限，这可能会导致网络性能不佳。本文提出了一种新的深度优化器，称为快速自适应矩估计（FAME），它首次使用三重指数移动平均值（TEMA）来估计梯度矩。将TEMA纳入优化过程中，可以提供更丰富和准确的数据变化和趋势信息，与目前所有主要自适应优化方法中使用的标准指数移动平均值相比。我们提出的FAME优化器已经在广泛的基准测试中得到了验证，包括CIFAR-10，CIFAR-100，PASCAL-VOC，MS-COCO和Cityscapes。

    Network optimization is a crucial step in the field of deep learning, as it directly affects the performance of models in various domains such as computer vision. Despite the numerous optimizers that have been developed over the years, the current methods are still limited in their ability to accurately and quickly identify gradient trends, which can lead to sub-optimal network performance. In this paper, we propose a novel deep optimizer called Fast-Adaptive Moment Estimation (FAME), which for the first time estimates gradient moments using a Triple Exponential Moving Average (TEMA). Incorporating TEMA into the optimization process provides richer and more accurate information on data changes and trends, as compared to the standard Exponential Moving Average used in essentially all current leading adaptive optimization methods. Our proposed FAME optimizer has been extensively validated through a wide range of benchmarks, including CIFAR-10, CIFAR-100, PASCAL-VOC, MS-COCO, and Cityscap
    

