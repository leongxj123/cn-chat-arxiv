# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [InkSight: Offline-to-Online Handwriting Conversion by Learning to Read and Write](https://arxiv.org/abs/2402.05804) | InkSight是一个可以将离线手写转换为在线手写的系统，通过结合阅读和书写先验知识，在多样化的照片中有效地Derendering手写文本。 |
| [^2] | [Strategic Client Selection to Address Non-IIDness in HAPS-enabled FL Networks.](http://arxiv.org/abs/2401.05308) | 该研究介绍了一种针对高空平台站（HAPS）使能的垂直异构网络中数据分布不均问题的战略客户选择策略，通过利用用户的网络流量行为预测和分类，优先选择数据呈现相似模式的客户参与，以提高联合学习（FL）模型的训练效果。 |
| [^3] | [Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures.](http://arxiv.org/abs/2307.15220) | 通过观看手术视频讲座，我们提出了一种新方法，SurgVLP，通过利用手术视频讲座中的语音和视觉信息进行多模态表示学习，并解决了手术相关语言挑战。 |
| [^4] | [FigCaps-HF: A Figure-to-Caption Generative Framework and Benchmark with Human Feedback.](http://arxiv.org/abs/2307.10867) | FigCaps-HF是一个图像生成标题的框架，可以通过融入领域专家的反馈意见，生成符合读者偏好的高质量图像标题。将自动评估和强化学习与人类反馈相结合，可以改善生成的标题与读者偏好的一致性。 |

# 详细

[^1]: InkSight：通过学习阅读和书写实现离线到在线手写转换

    InkSight: Offline-to-Online Handwriting Conversion by Learning to Read and Write

    [https://arxiv.org/abs/2402.05804](https://arxiv.org/abs/2402.05804)

    InkSight是一个可以将离线手写转换为在线手写的系统，通过结合阅读和书写先验知识，在多样化的照片中有效地Derendering手写文本。

    

    数字笔记正在变得越来越受欢迎，提供了一种耐用、可编辑和易于索引的存储笔记的方式，即矢量化形式的数字墨水。然而，这种笔记方式与传统的纸笔记方式之间仍存在显著差距，而传统纸笔记方式仍受到绝大多数人的青睐。我们的工作InkSight旨在弥合这种差距，使实体笔记者能够轻松地将他们的作品（离线手写）转换为数字墨水（在线手写），这个过程我们称之为Derendering。之前关于此主题的研究集中在图像的几何属性上，导致了在训练领域之外的有限泛化能力。我们的方法结合了阅读和书写的先验知识，允许在缺乏大量配对样本的情况下训练模型，而这些配对样本很难获取。据我们所知，这是第一个有效地对具有多样化视觉特征和背景的任意照片中的手写文本进行Derendering的工作。

    Digital note-taking is gaining popularity, offering a durable, editable, and easily indexable way of storing notes in the vectorized form, known as digital ink. However, a substantial gap remains between this way of note-taking and traditional pen-and-paper note-taking, a practice still favored by a vast majority. Our work, InkSight, aims to bridge the gap by empowering physical note-takers to effortlessly convert their work (offline handwriting) to digital ink (online handwriting), a process we refer to as Derendering. Prior research on the topic has focused on the geometric properties of images, resulting in limited generalization beyond their training domains. Our approach combines reading and writing priors, allowing training a model in the absence of large amounts of paired samples, which are difficult to obtain. To our knowledge, this is the first work that effectively derenders handwritten text in arbitrary photos with diverse visual characteristics and backgrounds. Furthermore,
    
[^2]: 面对HAPS使能的FL网络中的非独立同分布问题，战略客户选择的研究

    Strategic Client Selection to Address Non-IIDness in HAPS-enabled FL Networks. (arXiv:2401.05308v1 [cs.NI])

    [http://arxiv.org/abs/2401.05308](http://arxiv.org/abs/2401.05308)

    该研究介绍了一种针对高空平台站（HAPS）使能的垂直异构网络中数据分布不均问题的战略客户选择策略，通过利用用户的网络流量行为预测和分类，优先选择数据呈现相似模式的客户参与，以提高联合学习（FL）模型的训练效果。

    

    在由高空平台站（HAPS）使能的垂直异构网络中部署联合学习（FL）为各种不同通信和计算能力的客户提供了参与的机会。这种多样性不仅提高了FL模型的训练精度，还加快了其收敛速度。然而，在这些广阔的网络中应用FL存在显著的非独立同分布问题。这种数据异质性往往导致收敛速度较慢和模型训练性能的降低。我们的研究引入了一种针对此问题的客户选择策略，利用用户网络流量行为进行预测和分类。该策略通过战略性选择数据呈现相似模式的客户参与，同时优先考虑用户隐私。

    The deployment of federated learning (FL) within vertical heterogeneous networks, such as those enabled by high-altitude platform station (HAPS), offers the opportunity to engage a wide array of clients, each endowed with distinct communication and computational capabilities. This diversity not only enhances the training accuracy of FL models but also hastens their convergence. Yet, applying FL in these expansive networks presents notable challenges, particularly the significant non-IIDness in client data distributions. Such data heterogeneity often results in slower convergence rates and reduced effectiveness in model training performance. Our study introduces a client selection strategy tailored to address this issue, leveraging user network traffic behaviour. This strategy involves the prediction and classification of clients based on their network usage patterns while prioritizing user privacy. By strategically selecting clients whose data exhibit similar patterns for participation
    
[^3]: 通过观看数百个手术视频讲座学习多模态表示

    Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures. (arXiv:2307.15220v1 [cs.CV])

    [http://arxiv.org/abs/2307.15220](http://arxiv.org/abs/2307.15220)

    通过观看手术视频讲座，我们提出了一种新方法，SurgVLP，通过利用手术视频讲座中的语音和视觉信息进行多模态表示学习，并解决了手术相关语言挑战。

    

    最近在外科计算机视觉应用方面的进展主要依靠完全监督方法，主要使用视觉数据。这些方法依赖于手动注释的手术视频来预测一组固定的对象类别，限制了它们在未见手术程序和后续任务上的通用性。在这项工作中，我们提出了一个观点，即通过开放的手术电子学习平台提供的手术视频讲座可以为多模态表示学习提供有效的监督信号，而无需依赖手动注释。我们通过使用多个互补的自动语音识别系统生成文本转录来解决手术视频讲座中存在的手术相关语言挑战。然后，我们提出了一种新的方法，SurgVLP - 手术视觉语言预训练，用于多模态表示学习。SurgVLP构建了一种新的对比学习目标，将视频剪辑嵌入与相应的文本嵌入对齐。

    Recent advancements in surgical computer vision applications have been driven by fully-supervised methods, primarily using only visual data. These methods rely on manually annotated surgical videos to predict a fixed set of object categories, limiting their generalizability to unseen surgical procedures and downstream tasks. In this work, we put forward the idea that the surgical video lectures available through open surgical e-learning platforms can provide effective supervisory signals for multi-modal representation learning without relying on manual annotations. We address the surgery-specific linguistic challenges present in surgical video lectures by employing multiple complementary automatic speech recognition systems to generate text transcriptions. We then present a novel method, SurgVLP - Surgical Vision Language Pre-training, for multi-modal representation learning. SurgVLP constructs a new contrastive learning objective to align video clip embeddings with the corresponding m
    
[^4]: FigCaps-HF:一个基于人类反馈的图像生成标题框架和基准测试

    FigCaps-HF: A Figure-to-Caption Generative Framework and Benchmark with Human Feedback. (arXiv:2307.10867v1 [cs.CL])

    [http://arxiv.org/abs/2307.10867](http://arxiv.org/abs/2307.10867)

    FigCaps-HF是一个图像生成标题的框架，可以通过融入领域专家的反馈意见，生成符合读者偏好的高质量图像标题。将自动评估和强化学习与人类反馈相结合，可以改善生成的标题与读者偏好的一致性。

    

    标题对于理解科学可视化和文档至关重要。现有的科学图像生成标题方法依赖于从文档中提取的图像-标题配对进行训练，但其中许多配对在帮助性、解释性和视觉描述性等指标上存在不足，导致生成的标题与读者偏好不一致。为了能够生成高质量的图像标题，我们引入了FigCaps-HF，这是一个新的图像生成标题框架，可以融入领域专家的反馈意见，以生成优化了读者偏好的标题。我们的框架包含1）一种评估图像-标题配对质量的自动方法，2）一种基于人类反馈的强化学习（RLHF）方法，用于优化生成式图像生成标题模型以符合读者偏好。我们通过在不同类型的模型上改进性能，证明了我们简单的学习框架的有效性。

    Captions are crucial for understanding scientific visualizations and documents. Existing captioning methods for scientific figures rely on figure-caption pairs extracted from documents for training, many of which fall short with respect to metrics like helpfulness, explainability, and visual-descriptiveness [15] leading to generated captions being misaligned with reader preferences. To enable the generation of high-quality figure captions, we introduce FigCaps-HF a new framework for figure-caption generation that can incorporate domain expert feedback in generating captions optimized for reader preferences. Our framework comprises of 1) an automatic method for evaluating quality of figure-caption pairs, 2) a novel reinforcement learning with human feedback (RLHF) method to optimize a generative figure-to-caption model for reader preferences. We demonstrate the effectiveness of our simple learning framework by improving performance over standard fine-tuning across different types of mod
    

