# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AutoMMLab: Automatically Generating Deployable Models from Language Instructions for Computer Vision Tasks](https://arxiv.org/abs/2402.15351) | AutoMMLab是一个通用的LLM增强AutoML系统，通过用户的语言指令来自动化计算机视觉任务的整个模型生成工作流程，使非专家个体更容易构建特定任务的模型。 |
| [^2] | [Diffusion Model Based Visual Compensation Guidance and Visual Difference Analysis for No-Reference Image Quality Assessment](https://arxiv.org/abs/2402.14401) | 本研究将扩散模型引入无参考图像质量评估领域，设计了新的扩散恢复网络，提高了学习高级和低级视觉特征的效率。 |
| [^3] | [Pixel-Wise Recognition for Holistic Surgical Scene Understanding.](http://arxiv.org/abs/2401.11174) | 本文提出了一个整体和多粒度外科场景理解数据集，以及一个基于变形器的模型，该模型有效地结合了全局视频特征提取和局部器械分割，可用于多层次理解外科活动。 |
| [^4] | [PhotoBot: Reference-Guided Interactive Photography via Natural Language.](http://arxiv.org/abs/2401.11061) | PhotoBot是一个通过自然语言引导和机器人摄影师相互作用的自动化照片获取框架。它利用视觉语言模型和物体检测器来提供摄影建议，并通过视觉变换器计算相机的姿态调整，从而实现高质量的照片获取。 |
| [^5] | [Not All Steps are Equal: Efficient Generation with Progressive Diffusion Models.](http://arxiv.org/abs/2312.13307) | 提出了一种步骤自适应训练的两阶段策略，解决了传统扩散模型中训练过程中的冲突问题，将模型大小调整与噪声预测难度相匹配，提高了生成效果。 |
| [^6] | [A Comprehensive Augmentation Framework for Anomaly Detection.](http://arxiv.org/abs/2308.15068) | 本文提出了一种用于异常检测的综合增强框架，该框架通过选择性地利用适当的组合，分析并压缩模拟异常的关键特征，与基于重构的方法相结合，并采用分割训练策略，能够在MVTec异常检测数据集上优于以前的最先进方法。 |
| [^7] | [A Probabilistic Fluctuation based Membership Inference Attack for Generative Models.](http://arxiv.org/abs/2308.12143) | 本研究针对生成模型提出了一种概率波动评估成员推断攻击方法(PFAMI)，通过检测概率分布的波动性来推断模型中是否存在某条训练记录的成员身份。 |
| [^8] | [U-Turn Diffusion.](http://arxiv.org/abs/2308.07421) | U-Turn扩散是一种用于生成合成图像的AI模型，通过引入U-Turn Diffusion技术来改进生成图像的质量。这种技术结合了前向、U-Turn和反向过程，通过解构快速相关性来提高生成过程的效率。 |

# 详细

[^1]: AutoMMLab：从语言指令自动生成可部署模型用于计算机视觉任务

    AutoMMLab: Automatically Generating Deployable Models from Language Instructions for Computer Vision Tasks

    [https://arxiv.org/abs/2402.15351](https://arxiv.org/abs/2402.15351)

    AutoMMLab是一个通用的LLM增强AutoML系统，通过用户的语言指令来自动化计算机视觉任务的整个模型生成工作流程，使非专家个体更容易构建特定任务的模型。

    

    arXiv:2402.15351v1 公告类型：新 提要：自动化机器学习（AutoML）是一组旨在自动化机器学习开发过程的技术。虽然传统的AutoML方法已成功应用于模型开发的几个关键步骤（例如超参数优化），但缺乏一个可以自动化整个端到端模型生成工作流程的AutoML系统。为了填补这一空白，我们提出了AutoMMLab，这是一个通用的LLM增强AutoML系统，按照用户的语言指令来自动化计算机视觉任务的整个模型生成工作流程。所提出的AutoMMLab系统有效地利用LLM作为连接AutoML和OpenMMLab社区的桥梁，使非专家个体能够通过用户友好的语言界面轻松构建特定任务的模型。具体地，我们提出RU-LLaMA来理解用户的请求并安排整个流水线，并提出一种基于LLM的超参数优化器 c

    arXiv:2402.15351v1 Announce Type: new  Abstract: Automated machine learning (AutoML) is a collection of techniques designed to automate the machine learning development process. While traditional AutoML approaches have been successfully applied in several critical steps of model development (e.g. hyperparameter optimization), there lacks a AutoML system that automates the entire end-to-end model production workflow. To fill this blank, we present AutoMMLab, a general-purpose LLM-empowered AutoML system that follows user's language instructions to automate the whole model production workflow for computer vision tasks. The proposed AutoMMLab system effectively employs LLMs as the bridge to connect AutoML and OpenMMLab community, empowering non-expert individuals to easily build task-specific models via a user-friendly language interface. Specifically, we propose RU-LLaMA to understand users' request and schedule the whole pipeline, and propose a novel LLM-based hyperparameter optimizer c
    
[^2]: 基于扩散模型的视觉补偿引导和视觉差异分析用于无参考图像质量评估

    Diffusion Model Based Visual Compensation Guidance and Visual Difference Analysis for No-Reference Image Quality Assessment

    [https://arxiv.org/abs/2402.14401](https://arxiv.org/abs/2402.14401)

    本研究将扩散模型引入无参考图像质量评估领域，设计了新的扩散恢复网络，提高了学习高级和低级视觉特征的效率。

    

    现有的自由能引导的无参考图像质量评估(NR-IQA)方法仍然在找到在图像的像素级学习特征信息和捕获高级特征信息之间达到平衡以及高级特征信息的有效利用方面存在困难。作为一种新颖的领先技术(SOTA)生成模型类别，扩散模型展示了建模复杂关系的能力，能够全面理解图像，并具有更好地学习高级和低级视觉特征。鉴于此，我们首次将扩散模型探索到NR-IQA领域。首先，我们设计了一个新的扩散恢复网络，利用生成的增强图像和包含噪声的图像，将扩散模型去噪过程中获得的非线性特征作为高级视觉信息。

    arXiv:2402.14401v1 Announce Type: cross  Abstract: Existing free-energy guided No-Reference Image Quality Assessment (NR-IQA) methods still suffer from finding a balance between learning feature information at the pixel level of the image and capturing high-level feature information and the efficient utilization of the obtained high-level feature information remains a challenge. As a novel class of state-of-the-art (SOTA) generative model, the diffusion model exhibits the capability to model intricate relationships, enabling a comprehensive understanding of images and possessing a better learning of both high-level and low-level visual features. In view of these, we pioneer the exploration of the diffusion model into the domain of NR-IQA. Firstly, we devise a new diffusion restoration network that leverages the produced enhanced image and noise-containing images, incorporating nonlinear features obtained during the denoising process of the diffusion model, as high-level visual informat
    
[^3]: 像素级别识别用于整体外科场景理解

    Pixel-Wise Recognition for Holistic Surgical Scene Understanding. (arXiv:2401.11174v1 [cs.CV])

    [http://arxiv.org/abs/2401.11174](http://arxiv.org/abs/2401.11174)

    本文提出了一个整体和多粒度外科场景理解数据集，以及一个基于变形器的模型，该模型有效地结合了全局视频特征提取和局部器械分割，可用于多层次理解外科活动。

    

    本文提出了Prostatectomies的整体和多粒度外科场景理解（GraSP）数据集，该数据集对外科场景理解进行了层次化建模，包括不同粒度的互补任务。我们的方法实现了对外科活动的多层次理解，包括外科阶段和步骤的识别以及包括外科器械分割和原子可视动作检测在内的短期任务。为了利用我们提出的数据集，我们引入了基于变形器（Transformers）的行动、阶段、步骤和器械分割（TAPIS）模型，该模型将全局视频特征提取器与来自器械分割模型的局部区域建议相结合，以应对我们数据集的多粒度问题。通过广泛的实验，我们展示了在短期识别任务中包括分割注释的影响，并突显了不同的粒度要求。

    This paper presents the Holistic and Multi-Granular Surgical Scene Understanding of Prostatectomies (GraSP) dataset, a curated benchmark that models surgical scene understanding as a hierarchy of complementary tasks with varying levels of granularity. Our approach enables a multi-level comprehension of surgical activities, encompassing long-term tasks such as surgical phases and steps recognition and short-term tasks including surgical instrument segmentation and atomic visual actions detection. To exploit our proposed benchmark, we introduce the Transformers for Actions, Phases, Steps, and Instrument Segmentation (TAPIS) model, a general architecture that combines a global video feature extractor with localized region proposals from an instrument segmentation model to tackle the multi-granularity of our benchmark. Through extensive experimentation, we demonstrate the impact of including segmentation annotations in short-term recognition tasks, highlight the varying granularity require
    
[^4]: PhotoBot：通过自然语言引导的参考互动摄影

    PhotoBot: Reference-Guided Interactive Photography via Natural Language. (arXiv:2401.11061v1 [cs.CV])

    [http://arxiv.org/abs/2401.11061](http://arxiv.org/abs/2401.11061)

    PhotoBot是一个通过自然语言引导和机器人摄影师相互作用的自动化照片获取框架。它利用视觉语言模型和物体检测器来提供摄影建议，并通过视觉变换器计算相机的姿态调整，从而实现高质量的照片获取。

    

    我们介绍了一个名为PhotoBot的框架，它基于高级人类语言引导和机器人摄影师之间的相互作用，用于自动化的照片获取。我们建议通过从策展画廊中检索到的参考图片向用户传达摄影建议。我们利用视觉语言模型（VLM）和物体检测器，通过文本描述对参考图片进行特征化，并使用大型语言模型（LLM）通过基于用户语言查询的文本推理检索相关的参考图片。为了对应参考图片和观察到的场景，我们利用一个能够捕捉显著不同的图像的语义相似性的预训练特征的视觉变换器，通过解决一个透视n-点（PnP）问题来计算RGB-D相机的姿态调整。我们在配备有手腕相机的真实机械手臂上演示了我们的方法。我们的用户研究表明，由PhotoBot拍摄的照片具有良好的质量和效果。

    We introduce PhotoBot, a framework for automated photo acquisition based on an interplay between high-level human language guidance and a robot photographer. We propose to communicate photography suggestions to the user via a reference picture that is retrieved from a curated gallery. We exploit a visual language model (VLM) and an object detector to characterize reference pictures via textual descriptions and use a large language model (LLM) to retrieve relevant reference pictures based on a user's language query through text-based reasoning. To correspond the reference picture and the observed scene, we exploit pre-trained features from a vision transformer capable of capturing semantic similarity across significantly varying images. Using these features, we compute pose adjustments for an RGB-D camera by solving a Perspective-n-Point (PnP) problem. We demonstrate our approach on a real-world manipulator equipped with a wrist camera. Our user studies show that photos taken by PhotoBo
    
[^5]: 并非所有步骤都相等：进展扩散模型的高效生成

    Not All Steps are Equal: Efficient Generation with Progressive Diffusion Models. (arXiv:2312.13307v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.13307](http://arxiv.org/abs/2312.13307)

    提出了一种步骤自适应训练的两阶段策略，解决了传统扩散模型中训练过程中的冲突问题，将模型大小调整与噪声预测难度相匹配，提高了生成效果。

    

    扩散模型在多种生成任务中展示了出色的效能，具有去噪模型的预测能力。目前，这些模型在所有时间步上都采用统一的去噪方法。然而，每个时间步的噪声潜在变化导致了训练中的冲突，限制了扩散模型的潜力。为了解决这个挑战，我们提出了一种新的两阶段训练策略，称为步骤自适应训练。在初始阶段，训练一个基础的去噪模型来包括所有的时间步。随后，我们将时间步分为不同的组，对每个组内的模型进行微调，以达到专门的去噪能力。我们认识到，不同时间步的噪声预测困难程度是不同的，所以我们引入了多样的模型大小要求。我们通过估计每个时间步的信噪比来动态调整模型大小，以进行微调之前。此调整简化了模型的训练流程并提高了生成效果。

    Diffusion models have demonstrated remarkable efficacy in various generative tasks with the predictive prowess of denoising model. Currently, these models employ a uniform denoising approach across all timesteps. However, the inherent variations in noisy latents at each timestep lead to conflicts during training, constraining the potential of diffusion models. To address this challenge, we propose a novel two-stage training strategy termed Step-Adaptive Training. In the initial stage, a base denoising model is trained to encompass all timesteps. Subsequently, we partition the timesteps into distinct groups, fine-tuning the model within each group to achieve specialized denoising capabilities. Recognizing that the difficulties of predicting noise at different timesteps vary, we introduce a diverse model size requirement. We dynamically adjust the model size for each timestep by estimating task difficulty based on its signal-to-noise ratio before fine-tuning. This adjustment is facilitat
    
[^6]: 一种用于异常检测的综合增强框架

    A Comprehensive Augmentation Framework for Anomaly Detection. (arXiv:2308.15068v1 [cs.AI])

    [http://arxiv.org/abs/2308.15068](http://arxiv.org/abs/2308.15068)

    本文提出了一种用于异常检测的综合增强框架，该框架通过选择性地利用适当的组合，分析并压缩模拟异常的关键特征，与基于重构的方法相结合，并采用分割训练策略，能够在MVTec异常检测数据集上优于以前的最先进方法。

    

    数据增强方法通常被整合到异常检测模型的训练中。以往的方法主要集中在复制真实世界的异常或增加多样性，而没有考虑到异常的标准在不同类别之间存在差异，这可能导致训练分布的偏差。本文分析了对重构网络训练有贡献的模拟异常的关键特征，并将其压缩成几种方法，从而通过选择性地使用适当的组合来创建一个综合框架。此外，将这个框架与基于重构的方法相结合，并同时提出了一种分割训练策略，既减轻过拟合问题，又避免对重构过程引入干扰。在MVTec异常检测数据集上进行的评估表明，我们的方法在性能上优于以前的最先进方法，特别是在目标相关指标方面。

    Data augmentation methods are commonly integrated into the training of anomaly detection models. Previous approaches have primarily focused on replicating real-world anomalies or enhancing diversity, without considering that the standard of anomaly varies across different classes, potentially leading to a biased training distribution.This paper analyzes crucial traits of simulated anomalies that contribute to the training of reconstructive networks and condenses them into several methods, thus creating a comprehensive framework by selectively utilizing appropriate combinations.Furthermore, we integrate this framework with a reconstruction-based approach and concurrently propose a split training strategy that alleviates the issue of overfitting while avoiding introducing interference to the reconstruction process. The evaluations conducted on the MVTec anomaly detection dataset demonstrate that our method outperforms the previous state-of-the-art approach, particularly in terms of objec
    
[^7]: 一种基于概率波动的生成模型成员推断攻击方法

    A Probabilistic Fluctuation based Membership Inference Attack for Generative Models. (arXiv:2308.12143v1 [cs.LG])

    [http://arxiv.org/abs/2308.12143](http://arxiv.org/abs/2308.12143)

    本研究针对生成模型提出了一种概率波动评估成员推断攻击方法(PFAMI)，通过检测概率分布的波动性来推断模型中是否存在某条训练记录的成员身份。

    

    成员推断攻击(MIA)通过查询模型来识别机器学习模型的训练集中是否存在某条记录。对经典分类模型的MIA已有很多研究，最近的工作开始探索如何将MIA应用到生成模型上。我们的研究表明，现有的面向生成模型的MIA主要依赖于目标模型的过拟合现象。然而，过拟合可以通过采用各种正则化技术来避免，而现有的MIA在实践中表现不佳。与过拟合不同，记忆对于深度学习模型实现最佳性能是至关重要的，使其成为一种更为普遍的现象。生成模型中的记忆导致生成记录的概率分布呈现出增长的趋势。因此，我们提出了一种基于概率波动的成员推断攻击方法(PFAMI)，它是一种黑盒MIA，通过检测概率波动来推断成员身份。

    Membership Inference Attack (MIA) identifies whether a record exists in a machine learning model's training set by querying the model. MIAs on the classic classification models have been well-studied, and recent works have started to explore how to transplant MIA onto generative models. Our investigation indicates that existing MIAs designed for generative models mainly depend on the overfitting in target models. However, overfitting can be avoided by employing various regularization techniques, whereas existing MIAs demonstrate poor performance in practice. Unlike overfitting, memorization is essential for deep learning models to attain optimal performance, making it a more prevalent phenomenon. Memorization in generative models leads to an increasing trend in the probability distribution of generating records around the member record. Therefore, we propose a Probabilistic Fluctuation Assessing Membership Inference Attack (PFAMI), a black-box MIA that infers memberships by detecting t
    
[^8]: U-Turn扩散

    U-Turn Diffusion. (arXiv:2308.07421v1 [cs.LG])

    [http://arxiv.org/abs/2308.07421](http://arxiv.org/abs/2308.07421)

    U-Turn扩散是一种用于生成合成图像的AI模型，通过引入U-Turn Diffusion技术来改进生成图像的质量。这种技术结合了前向、U-Turn和反向过程，通过解构快速相关性来提高生成过程的效率。

    

    我们对基于分数的扩散模型进行了全面研究，用于生成合成图像的AI模型。这些模型依赖于由随机微分方程驱动的动态辅助时间机制，在输入图像中获取分数函数。我们的研究揭示了评估基于分数的扩散模型效率的标准：生成过程的能力取决于在反向/去噪阶段解构快速相关性的能力。为了提高生成的合成图像质量，我们引入了一种被称为“U-Turn Diffusion”的方法。U-Turn Diffusion技术从标准的前向扩散过程开始，尽管相对于传统设置，它的持续时间更短。随后，我们执行标准的反向动力学，以前向过程的最终配置为初始值。这种结合了前向、U-Turn和反向过程的U-Turn Diffusion过程创建一个合成图像。

    We present a comprehensive examination of score-based diffusion models of AI for generating synthetic images. These models hinge upon a dynamic auxiliary time mechanism driven by stochastic differential equations, wherein the score function is acquired from input images. Our investigation unveils a criterion for evaluating efficiency of the score-based diffusion models: the power of the generative process depends on the ability to de-construct fast correlations during the reverse/de-noising phase. To improve the quality of the produced synthetic images, we introduce an approach coined "U-Turn Diffusion". The U-Turn Diffusion technique starts with the standard forward diffusion process, albeit with a condensed duration compared to conventional settings. Subsequently, we execute the standard reverse dynamics, initialized with the concluding configuration from the forward process. This U-Turn Diffusion procedure, combining forward, U-turn, and reverse processes, creates a synthetic image 
    

