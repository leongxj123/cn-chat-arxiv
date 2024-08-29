# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DeepMIF: Deep Monotonic Implicit Fields for Large-Scale LiDAR 3D Mapping](https://arxiv.org/abs/2403.17550) | 提出了DeepMIF，通过设计学习系统集成单调性损失，在大规模3D地图绘制中优化神经单调场，避免了LiDAR测量的嘈杂问题 |
| [^2] | [MolNexTR: A Generalized Deep Learning Model for Molecular Image Recognition](https://arxiv.org/abs/2403.03691) | MolNexTR是一种用于分子图像识别的通用深度学习模型，能够更细致提取分子图像的局部和全局特征，同时能够预测原子和键，理解布局规则，灵活整合符号化的化学原则，并且包含多种先进算法。 |
| [^3] | [Examining Pathological Bias in a Generative Adversarial Network Discriminator: A Case Study on a StyleGAN3 Model](https://arxiv.org/abs/2402.09786) | 这项研究发现了StyleGAN3模型中判别器的病态偏见，它在图像和面部质量上的得分分层影响了不同性别、种族和其他类别的图像。 |
| [^4] | [Solid Waste Detection in Remote Sensing Images: A Survey](https://arxiv.org/abs/2402.09066) | 本文调查了固体废物在遥感图像中的检测方法。研究者利用地球观测卫星提供的高分辨率数据，通过遥感图像实现了固体废物处置场地的识别、监测和评估。 |
| [^5] | [FERGI: Automatic Annotation of User Preferences for Text-to-Image Generation from Spontaneous Facial Expression Reaction](https://arxiv.org/abs/2312.03187) | 开发了一种从用户自发面部表情反应中自动注释用户对生成图像偏好的方法，发现多个面部动作单元与用户对生成图像的评估高度相关，可用于通过这些面部动作单元区分图像对并自动标注用户偏好。 |
| [^6] | [Provable Probabilistic Imaging using Score-Based Generative Priors.](http://arxiv.org/abs/2310.10835) | 本文提出了一种基于得分的生成先验的插入式蒙特卡洛算法，能够实现高质量图像重建和不确定性量化。 |
| [^7] | [When Multi-Task Learning Meets Partial Supervision: A Computer Vision Review.](http://arxiv.org/abs/2307.14382) | 本综述讨论了多任务学习如何在部分监督设置下应用，以解决由于复杂的优化方案和高标签需求而引入的挑战。 |
| [^8] | [Expert Knowledge-Aware Image Difference Graph Representation Learning for Difference-Aware Medical Visual Question Answering.](http://arxiv.org/abs/2307.11986) | 本研究提出了一个新的医学视觉问答任务，名为MIMIC-Diff-VQA，为自动化医学视觉语言模型做出了贡献。与现有数据集相比，该任务旨在回答关于疾病和图像差异的问题，并应用了专家知识感知的图表示学习模型。 |
| [^9] | [Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC.](http://arxiv.org/abs/2302.11552) | 该论文提出了一种基于能量扩散模型和MCMC的组合生成方法，旨在解决现有技术在组合生成中的失败问题，并提出了新的成功的解决方案。 |
| [^10] | [Multimodality Helps Unimodality: Cross-Modal Few-Shot Learning with Multimodal Models.](http://arxiv.org/abs/2301.06267) | 通过跨模态适应方法，在多模态模型下利用少样本示例（包括文本和声音）进行狗的视觉分类，并取得了最先进的结果。 |

# 详细

[^1]: DeepMIF: 用于大规模LiDAR 3D地图绘制的深度单调隐式场

    DeepMIF: Deep Monotonic Implicit Fields for Large-Scale LiDAR 3D Mapping

    [https://arxiv.org/abs/2403.17550](https://arxiv.org/abs/2403.17550)

    提出了DeepMIF，通过设计学习系统集成单调性损失，在大规模3D地图绘制中优化神经单调场，避免了LiDAR测量的嘈杂问题

    

    近年来，通过使用现代获取设备如LiDAR传感器，在感知真实大规模室外3D环境方面取得了显著进展。然而，它们在生成稠密、完整的3D场景方面存在固有限制。为解决这一问题，最近的基于学习的方法集成了神经隐式表示和可优化特征网格，以逼近3D场景的表面。然而，简单地沿原始LiDAR光线拟合样本会导致由于稀疏、互相矛盾的LiDAR测量的特性而产生嘈杂的3D绘图结果。相反，在这项工作中，我们不再精确拟合LiDAR数据，而是让网络优化在3D空间中定义的非度量单调隐式场。为适应我们的场，我们设计了一个学习系统，集成了一个单调性损失，使得能够优化神经单调场并利用了大规模3D地图绘制的最新进展。我们的算法...

    arXiv:2403.17550v1 Announce Type: cross  Abstract: Recently, significant progress has been achieved in sensing real large-scale outdoor 3D environments, particularly by using modern acquisition equipment such as LiDAR sensors. Unfortunately, they are fundamentally limited in their ability to produce dense, complete 3D scenes. To address this issue, recent learning-based methods integrate neural implicit representations and optimizable feature grids to approximate surfaces of 3D scenes. However, naively fitting samples along raw LiDAR rays leads to noisy 3D mapping results due to the nature of sparse, conflicting LiDAR measurements. Instead, in this work we depart from fitting LiDAR data exactly, instead letting the network optimize a non-metric monotonic implicit field defined in 3D space. To fit our field, we design a learning system integrating a monotonicity loss that enables optimizing neural monotonic fields and leverages recent progress in large-scale 3D mapping. Our algorithm ac
    
[^2]: MolNexTR：一种用于分子图像识别的通用深度学习模型

    MolNexTR: A Generalized Deep Learning Model for Molecular Image Recognition

    [https://arxiv.org/abs/2403.03691](https://arxiv.org/abs/2403.03691)

    MolNexTR是一种用于分子图像识别的通用深度学习模型，能够更细致提取分子图像的局部和全局特征，同时能够预测原子和键，理解布局规则，灵活整合符号化的化学原则，并且包含多种先进算法。

    

    在化学结构识别领域，将分子图像转换为图结构和SMILES字符串的任务是一个重要挑战，主要是由于化学文献中流行的各种绘图风格和约定。为了弥合这一差距，我们提出了MolNexTR，一种新颖的图像到图结构的深度学习模型，它合并了ConvNext和Vision-TRansformer的优势，实现了对分子图像中的局部和全局特征的更细致提取。MolNexTR可以同时预测原子和键，并理解它们的布局规则。它还擅长灵活地将符号化的化学原则融入其中，以识别手性并解析缩写结构。我们进一步整合了一系列先进算法，包括改进的数据增强模块、图像污染模块和后处理模块。

    arXiv:2403.03691v1 Announce Type: cross  Abstract: In the field of chemical structure recognition, the task of converting molecular images into graph structures and SMILES string stands as a significant challenge, primarily due to the varied drawing styles and conventions prevalent in chemical literature. To bridge this gap, we proposed MolNexTR, a novel image-to-graph deep learning model that collaborates to fuse the strengths of ConvNext, a powerful Convolutional Neural Network variant, and Vision-TRansformer. This integration facilitates a more nuanced extraction of both local and global features from molecular images. MolNexTR can predict atoms and bonds simultaneously and understand their layout rules. It also excels at flexibly integrating symbolic chemistry principles to discern chirality and decipher abbreviated structures. We further incorporate a series of advanced algorithms, including improved data augmentation module, image contamination module, and a post-processing modul
    
[^3]: 检查生成对抗网络判别器中的病态偏见：以StyleGAN3模型为例的案例研究

    Examining Pathological Bias in a Generative Adversarial Network Discriminator: A Case Study on a StyleGAN3 Model

    [https://arxiv.org/abs/2402.09786](https://arxiv.org/abs/2402.09786)

    这项研究发现了StyleGAN3模型中判别器的病态偏见，它在图像和面部质量上的得分分层影响了不同性别、种族和其他类别的图像。

    

    生成对抗网络可以生成逼真的人脸，往往难以被人类区分出来。我们发现预训练的StyleGAN3模型中的判别器在图像和面部质量上系统地对得分进行分层，并且这不成比例地影响了不同性别、种族和其他类别的图像。我们检查了判别器在色彩和亮度方面对感知的种族和性别的偏见，然后检查了社会心理学中关于刻板印象研究中常见的偏见。

    arXiv:2402.09786v1 Announce Type: cross  Abstract: Generative adversarial networks generate photorealistic faces that are often indistinguishable by humans from real faces. We find that the discriminator in the pre-trained StyleGAN3 model, a popular GAN network, systematically stratifies scores by both image- and face-level qualities and that this disproportionately affects images across gender, race, and other categories. We examine the discriminator's bias for color and luminance across axes perceived race and gender; we then examine axes common in research on stereotyping in social psychology.
    
[^4]: 遥感图像中的固体废物检测：一项调查

    Solid Waste Detection in Remote Sensing Images: A Survey

    [https://arxiv.org/abs/2402.09066](https://arxiv.org/abs/2402.09066)

    本文调查了固体废物在遥感图像中的检测方法。研究者利用地球观测卫星提供的高分辨率数据，通过遥感图像实现了固体废物处置场地的识别、监测和评估。

    

    识别和表征非法固体废物处置场地对环境保护至关重要，特别是应对污染和健康危害。不当管理的垃圾填埋场通过雨水渗透污染土壤和地下水，对动物和人类构成威胁。传统的填埋场辨识方法，如现场检查，耗时且昂贵。遥感技术是用于识别和监测固体废物处置场地的一种经济有效的解决方案，可以实现广泛覆盖和多次获取。地球观测（EO）卫星配备了一系列传感器和成像能力，几十年来一直提供高分辨率的数据。研究人员提出了专门的技术，利用遥感图像执行一系列任务，如废物场地检测、倾倒场监测和适宜位置评估。

    arXiv:2402.09066v1 Announce Type: cross Abstract: The detection and characterization of illegal solid waste disposal sites are essential for environmental protection, particularly for mitigating pollution and health hazards. Improperly managed landfills contaminate soil and groundwater via rainwater infiltration, posing threats to both animals and humans. Traditional landfill identification approaches, such as on-site inspections, are time-consuming and expensive. Remote sensing is a cost-effective solution for the identification and monitoring of solid waste disposal sites that enables broad coverage and repeated acquisitions over time. Earth Observation (EO) satellites, equipped with an array of sensors and imaging capabilities, have been providing high-resolution data for several decades. Researchers proposed specialized techniques that leverage remote sensing imagery to perform a range of tasks such as waste site detection, dumping site monitoring, and assessment of suitable locati
    
[^5]: FERGI：来自自发面部表情反应的文本到图像生成用户偏好的自动注释

    FERGI: Automatic Annotation of User Preferences for Text-to-Image Generation from Spontaneous Facial Expression Reaction

    [https://arxiv.org/abs/2312.03187](https://arxiv.org/abs/2312.03187)

    开发了一种从用户自发面部表情反应中自动注释用户对生成图像偏好的方法，发现多个面部动作单元与用户对生成图像的评估高度相关，可用于通过这些面部动作单元区分图像对并自动标注用户偏好。

    

    研究人员提出使用人类偏好反馈数据来微调文本到图像生成模型。然而，由于其依赖于手动注释，人类反馈收集的可扩展性受到限制。因此，我们开发并测试了一种方法，从用户的自发面部表情反应中自动注释其对生成图像的偏好。我们收集了一个面部表情反应到生成图像（FERGI）的数据集，并展示了多个面部运动单元（AUs）的激活与用户对生成图像的评估高度相关。具体来说，AU4（眉毛下垂者）反映了对生成图像的负面评价，而AU12（嘴角拉动者）反映了正面评价。这两者在两个方面都很有用。首先，我们可以准确地使用这些AU响应存在实质差异的图像对之间自动注释用户偏好。

    arXiv:2312.03187v2 Announce Type: replace-cross  Abstract: Researchers have proposed to use data of human preference feedback to fine-tune text-to-image generative models. However, the scalability of human feedback collection has been limited by its reliance on manual annotation. Therefore, we develop and test a method to automatically annotate user preferences from their spontaneous facial expression reaction to the generated images. We collect a dataset of Facial Expression Reaction to Generated Images (FERGI) and show that the activations of multiple facial action units (AUs) are highly correlated with user evaluations of the generated images. Specifically, AU4 (brow lowerer) is reflective of negative evaluations of the generated image whereas AU12 (lip corner puller) is reflective of positive evaluations. These can be useful in two ways. Firstly, we can automatically annotate user preferences between image pairs with substantial difference in these AU responses with an accuracy sig
    
[^6]: 用基于得分的生成先验的可证明的概率成像

    Provable Probabilistic Imaging using Score-Based Generative Priors. (arXiv:2310.10835v1 [eess.IV])

    [http://arxiv.org/abs/2310.10835](http://arxiv.org/abs/2310.10835)

    本文提出了一种基于得分的生成先验的插入式蒙特卡洛算法，能够实现高质量图像重建和不确定性量化。

    

    在解决反问题时，估计高质量图像并量化其不确定性是图像重建算法中的两个理想特点。本文提出了插入式蒙特卡洛（PMC）作为一种对一般反问题可能解空间进行建模的原则性框架。PMC能够通过后验采样来结合丰富的基于得分的生成先验进行高质量图像重建，并进行不确定性量化。具体而言，我们引入了两种PMC算法，可以视为传统插入式先验（PnP）和去噪正则化（RED）算法的采样模拟。我们还建立了对PMC算法收敛性的理论分析。我们的分析为两种算法提供了非渐近稳定性保证，即使在非对数凹似然和不完美得分网络的情况下也是如此。

    Estimating high-quality images while also quantifying their uncertainty are two desired features in an image reconstruction algorithm for solving ill-posed inverse problems. In this paper, we propose plug-and-play Monte Carlo (PMC) as a principled framework for characterizing the space of possible solutions to a general inverse problem. PMC is able to incorporate expressive score-based generative priors for high-quality image reconstruction while also performing uncertainty quantification via posterior sampling. In particular, we introduce two PMC algorithms which can be viewed as the sampling analogues of the traditional plug-and-play priors (PnP) and regularization by denoising (RED) algorithms. We also establish a theoretical analysis for characterizing the convergence of the PMC algorithms. Our analysis provides non-asymptotic stationarity guarantees for both algorithms, even in the presence of non-log-concave likelihoods and imperfect score networks. We demonstrate the performance
    
[^7]: 当多任务学习遇到部分监督：计算机视觉综述

    When Multi-Task Learning Meets Partial Supervision: A Computer Vision Review. (arXiv:2307.14382v1 [cs.LG])

    [http://arxiv.org/abs/2307.14382](http://arxiv.org/abs/2307.14382)

    本综述讨论了多任务学习如何在部分监督设置下应用，以解决由于复杂的优化方案和高标签需求而引入的挑战。

    

    多任务学习(MTL)旨在同时学习多个任务，并利用它们之间的相互关系。通过使用共享资源同时计算多个输出，这种学习范式有潜力比传统方法在内存需求和推理时间方面更低。以往的MTL研究主要集中在完全监督方法上，因为任务之间的关系可以降低这些方法对数据的依赖性，并且可以提高性能。然而，MTL引入了一系列挑战，由于复杂的优化方案和更高的标签需求。本综述着重于MTL如何在不同的部分监督设置下应用，以解决这些挑战。首先，本综述分析了MTL传统上如何使用不同的参数共享技术在任务之间进行知识转移。其次，它介绍了不同的挑战。

    Multi-Task Learning (MTL) aims to learn multiple tasks simultaneously while exploiting their mutual relationships. By using shared resources to simultaneously calculate multiple outputs, this learning paradigm has the potential to have lower memory requirements and inference times compared to the traditional approach of using separate methods for each task. Previous work in MTL has mainly focused on fully-supervised methods, as task relationships can not only be leveraged to lower the level of data-dependency of those methods but they can also improve performance. However, MTL introduces a set of challenges due to a complex optimisation scheme and a higher labeling requirement. This review focuses on how MTL could be utilised under different partial supervision settings to address these challenges. First, this review analyses how MTL traditionally uses different parameter sharing techniques to transfer knowledge in between tasks. Second, it presents the different challenges arising fro
    
[^8]: Expert Knowledge-Aware Image Difference Graph Representation Learning for Difference-Aware Medical Visual Question Answering.（专家知识感知的图像变化图表示学习用于关注差异的医学视觉问答）

    Expert Knowledge-Aware Image Difference Graph Representation Learning for Difference-Aware Medical Visual Question Answering. (arXiv:2307.11986v1 [cs.CV])

    [http://arxiv.org/abs/2307.11986](http://arxiv.org/abs/2307.11986)

    本研究提出了一个新的医学视觉问答任务，名为MIMIC-Diff-VQA，为自动化医学视觉语言模型做出了贡献。与现有数据集相比，该任务旨在回答关于疾病和图像差异的问题，并应用了专家知识感知的图表示学习模型。

    

    为了为自动化医学视觉语言模型做出贡献，我们提出了一个新颖的胸部X光图像差异视觉问答（VQA）任务。该任务旨在回答几个关于疾病以及更重要的是它们之间差异的问题。这与放射科医生的诊断实践相一致，放射科医生在得出报告之前会对当前图像与参考图像进行比较。我们收集了一个新的数据集，称为MIMIC-Diff-VQA，包括来自164,324对主图像和参考图像的700,703个问题-答案配对。与现有的医学VQA数据集相比，我们的问题针对了临床专业人员使用的评估-诊断-干预-评估治疗过程。同时，我们还提出了一种新的专家知识感知的图表示学习模型来解决这个任务。

    To contribute to automating the medical vision-language model, we propose a novel Chest-Xray Difference Visual Question Answering (VQA) task. Given a pair of main and reference images, this task attempts to answer several questions on both diseases and, more importantly, the differences between them. This is consistent with the radiologist's diagnosis practice that compares the current image with the reference before concluding the report. We collect a new dataset, namely MIMIC-Diff-VQA, including 700,703 QA pairs from 164,324 pairs of main and reference images. Compared to existing medical VQA datasets, our questions are tailored to the Assessment-Diagnosis-Intervention-Evaluation treatment procedure used by clinical professionals. Meanwhile, we also propose a novel expert knowledge-aware graph representation learning model to address this task. The proposed baseline model leverages expert knowledge such as anatomical structure prior, semantic, and spatial knowledge to construct a mul
    
[^9]: 减少、重复利用、回收：基于能量扩散模型和MCMC的组合生成

    Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC. (arXiv:2302.11552v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.11552](http://arxiv.org/abs/2302.11552)

    该论文提出了一种基于能量扩散模型和MCMC的组合生成方法，旨在解决现有技术在组合生成中的失败问题，并提出了新的成功的解决方案。

    

    自从扩散模型问世以来，它在许多领域中已经迅速成为生成模型的主要方法。它们可以被解释为学习一系列时变的对数概率密度函数的梯度。这种解释已经激发了基于分类器和无分类器指导的思想成为后续控制扩散模型的方法。在这项工作中，我们建立在这些想法的基础上，利用扩散模型的分数-based解释，探索了用于涉及组合生成和指导的条件、修改和重复使用扩散模型的替代方法。特别是，我们调查了为什么某些类型的组合使用当前技术失败，并介绍了一些解决方案。我们得出结论，采样者(而不是模型)对此失败负有责任，并提出了新的采样器，受MCMC的启发，使组合生成成功。此外，我们提出了一种基于能量的扩散模型参数化方法，它使得逼近目标分布更加容易。

    Since their introduction, diffusion models have quickly become the prevailing approach to generative modeling in many domains. They can be interpreted as learning the gradients of a time-varying sequence of log-probability density functions. This interpretation has motivated classifier-based and classifier-free guidance as methods for post-hoc control of diffusion models. In this work, we build upon these ideas using the score-based interpretation of diffusion models, and explore alternative ways to condition, modify, and reuse diffusion models for tasks involving compositional generation and guidance. In particular, we investigate why certain types of composition fail using current techniques and present a number of solutions. We conclude that the sampler (not the model) is responsible for this failure and propose new samplers, inspired by MCMC, which enable successful compositional generation. Further, we propose an energy-based parameterization of diffusion models which enables the 
    
[^10]: 多模态有助于单模态：多模态模型下的交叉模态少样本学习

    Multimodality Helps Unimodality: Cross-Modal Few-Shot Learning with Multimodal Models. (arXiv:2301.06267v4 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2301.06267](http://arxiv.org/abs/2301.06267)

    通过跨模态适应方法，在多模态模型下利用少样本示例（包括文本和声音）进行狗的视觉分类，并取得了最先进的结果。

    

    快速学习新任务的能力是智能代理的核心要素，也被称为少样本学习。传统的少样本学习基准使用来自单模态的少样本样本，但这些样本可能不足以描述整个概念类。相比之下，人类使用跨模态信息高效地学习新概念。在这项工作中，我们展示了通过阅读关于狗并听它们吠叫的声音来构建更好的视觉狗分类器的可能性。为此，我们利用最近的多模态基础模型（如CLIP）是固有的跨模态的特性，将不同的模态映射到相同的表示空间。具体而言，我们提出了一种简单的跨模态适应方法，从跨越不同模态的少样本示例中进行学习。通过将类名重新用作额外的一次性训练样本，我们使用一个极其简单的线性分类器实现了最先进的结果。

    The ability to quickly learn a new task with minimal instruction - known as few-shot learning - is a central aspect of intelligent agents. Classical few-shot benchmarks make use of few-shot samples from a single modality, but such samples may not be sufficient to characterize an entire concept class. In contrast, humans use cross-modal information to learn new concepts efficiently. In this work, we demonstrate that one can indeed build a better ${\bf visual}$ dog classifier by ${\bf read}$ing about dogs and ${\bf listen}$ing to them bark. To do so, we exploit the fact that recent multimodal foundation models such as CLIP are inherently cross-modal, mapping different modalities to the same representation space. Specifically, we propose a simple cross-modal adaptation approach that learns from few-shot examples spanning different modalities. By repurposing class names as additional one-shot training samples, we achieve SOTA results with an embarrassingly simple linear classifier for visi
    

