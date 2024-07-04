# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Accelerating Diffusion Sampling with Optimized Time Steps](https://arxiv.org/abs/2402.17376) | 提出了一个通用框架用于设计优化问题，旨在通过寻找更合适的时间步长加速扩散采样。 |
| [^2] | [Explainable AI for Safe and Trustworthy Autonomous Driving: A Systematic Review](https://arxiv.org/abs/2402.10086) | 可解释的AI技术对于解决自动驾驶中的安全问题和信任问题至关重要。本文通过系统文献综述的方式，分析了可解释的AI方法在满足自动驾驶要求方面的关键贡献，并提出了可解释的设计、可解释的替代模型、可解释的监控、辅助技术和解释的可视化等五个方面的应用。 |
| [^3] | [Organic or Diffused: Can We Distinguish Human Art from AI-generated Images?](https://arxiv.org/abs/2402.03214) | 这项研究探讨了如何区分人类艺术和AI生成的图像，并提供了几种不同的方法，包括通过监督学习训练的分类器、扩散模型的研究工具以及专业艺术家的知识。这对防止欺诈、遵守政策以及避免模型崩溃都具有重要意义。 |
| [^4] | [RIDGE: Reproducibility, Integrity, Dependability, Generalizability, and Efficiency Assessment of Medical Image Segmentation Models.](http://arxiv.org/abs/2401.08847) | RIDGE是一个用于评估医学图像分割模型的可重复性、完整性、可靠性、泛化性和效率的框架，旨在通过提高工作质量和透明度，确保分割模型在科学可靠性和临床相关性上都具备优势。 |
| [^5] | [Mixture-of-Experts for Open Set Domain Adaptation: A Dual-Space Detection Approach.](http://arxiv.org/abs/2311.00285) | 该论文提出了一种Mixture-of-Experts用于开放域适应的双空间检测方法，利用图像特征空间和路由特征空间之间的不一致性来检测未知类别的样本，无需手动调节阈值。 |
| [^6] | [Three Ways to Improve Verbo-visual Fusion for Dense 3D Visual Grounding.](http://arxiv.org/abs/2309.04561) | 提出了一个稠密三维引用网络ConcreteNet，包含三个新模块，旨在改善具有相同语义类别干扰因素的重复实例的引用性能。 |
| [^7] | [Semi-Supervised Semantic Segmentation via Marginal Contextual Information.](http://arxiv.org/abs/2308.13900) | 通过利用分割图中标签的空间相关性，我们提出的S4MC方法在半监督语义分割中通过增强伪标签的方式，并提高了无标签数据的使用量，从而实现了超越现有方法的性能提升。 |
| [^8] | [How Deep Neural Networks Learn Compositional Data: The Random Hierarchy Model.](http://arxiv.org/abs/2307.02129) | 本文研究了深度神经网络学习组合性数据的问题，通过对随机层次模型进行分类任务，发现深度CNN学习这个任务所需的训练数据数量随着类别数、组合数和迭代次数的增加而渐进增加。 |
| [^9] | [A Framework For Refining Text Classification and Object Recognition from Academic Articles.](http://arxiv.org/abs/2305.17401) | 本文提出了一种结合基于规则的方法和机器学习的框架，旨在解决从学术论文中提炼文本分类和对象识别的问题。 |
| [^10] | [Large-scale Pre-trained Models are Surprisingly Strong in Incremental Novel Class Discovery.](http://arxiv.org/abs/2303.15975) | 本论文提出了一种更加挑战性和实用性的学习方法MSc-iNCD，通过在连续而无人监督的学习中利用大规模预训练模型的丰富先验知识，该方法在增量式新类别发现中表现出出乎意料的强大实力。 |
| [^11] | [A Systematic Performance Analysis of Deep Perceptual Loss Networks: Breaking Transfer Learning Conventions.](http://arxiv.org/abs/2302.04032) | 这项工作通过系统评估多种常用的预训练网络及其不同特征提取点，在四个深度感知损失用例上解决了迁移学习中的问题。 |

# 详细

[^1]: 优化时间步长加速扩散采样

    Accelerating Diffusion Sampling with Optimized Time Steps

    [https://arxiv.org/abs/2402.17376](https://arxiv.org/abs/2402.17376)

    提出了一个通用框架用于设计优化问题，旨在通过寻找更合适的时间步长加速扩散采样。

    

    扩散概率模型（DPMs）在高分辨率图像合成中表现出色，但由于通常需要大量采样步骤，其采样效率仍有待提高。近期高阶数值ODE求解器在DPMs中的应用使得用更少的采样步骤生成高质量图像成为可能。尽管这是一项重大进展，大多数采样方法仍然采用均匀时间步长，而在采样步骤较少时并不是最佳选择。为解决这一问题，我们提出了一个通用框架，用于设计一个优化问题，该优化问题旨在为DPMs的特定数值ODE求解器寻找更合适的时间步长。此优化问题旨在最小化地实现地真实解与与数值求解器对应的近似解之间的距离。它可以通过受限信赖域方法进行高效求解，时间少于

    arXiv:2402.17376v1 Announce Type: cross  Abstract: Diffusion probabilistic models (DPMs) have shown remarkable performance in high-resolution image synthesis, but their sampling efficiency is still to be desired due to the typically large number of sampling steps. Recent advancements in high-order numerical ODE solvers for DPMs have enabled the generation of high-quality images with much fewer sampling steps. While this is a significant development, most sampling methods still employ uniform time steps, which is not optimal when using a small number of steps. To address this issue, we propose a general framework for designing an optimization problem that seeks more appropriate time steps for a specific numerical ODE solver for DPMs. This optimization problem aims to minimize the distance between the ground-truth solution to the ODE and an approximate solution corresponding to the numerical solver. It can be efficiently solved using the constrained trust region method, taking less than 
    
[^2]: 可解释的人工智能在安全可信的自动驾驶中的应用：一项系统性评述

    Explainable AI for Safe and Trustworthy Autonomous Driving: A Systematic Review

    [https://arxiv.org/abs/2402.10086](https://arxiv.org/abs/2402.10086)

    可解释的AI技术对于解决自动驾驶中的安全问题和信任问题至关重要。本文通过系统文献综述的方式，分析了可解释的AI方法在满足自动驾驶要求方面的关键贡献，并提出了可解释的设计、可解释的替代模型、可解释的监控、辅助技术和解释的可视化等五个方面的应用。

    

    鉴于其在感知和规划任务中相对传统方法具有更优异的性能，人工智能（AI）对于自动驾驶（AD）的应用显示出了很大的潜力。然而，难以理解的AI系统加剧了对AD安全保证的现有挑战。缓解这一挑战的一种方法是利用可解释的AI（XAI）技术。为此，我们首次提出了关于可解释方法在安全可信的AD中的全面系统文献综述。我们首先分析了在AD背景下AI的要求，重点关注数据、模型和机构这三个关键方面。我们发现XAI对于满足这些要求是至关重要的。基于此，我们解释了AI中解释的来源，并描述了一种XAI的分类学。然后，我们确定了XAI在安全可信的AD中的五个主要贡献，包括可解释的设计、可解释的替代模型、可解释的监控，辅助...

    arXiv:2402.10086v1 Announce Type: cross  Abstract: Artificial Intelligence (AI) shows promising applications for the perception and planning tasks in autonomous driving (AD) due to its superior performance compared to conventional methods. However, inscrutable AI systems exacerbate the existing challenge of safety assurance of AD. One way to mitigate this challenge is to utilize explainable AI (XAI) techniques. To this end, we present the first comprehensive systematic literature review of explainable methods for safe and trustworthy AD. We begin by analyzing the requirements for AI in the context of AD, focusing on three key aspects: data, model, and agency. We find that XAI is fundamental to meeting these requirements. Based on this, we explain the sources of explanations in AI and describe a taxonomy of XAI. We then identify five key contributions of XAI for safe and trustworthy AI in AD, which are interpretable design, interpretable surrogate models, interpretable monitoring, auxil
    
[^3]: 有机或扩散：我们能区分人类艺术和AI生成的图像吗？

    Organic or Diffused: Can We Distinguish Human Art from AI-generated Images?

    [https://arxiv.org/abs/2402.03214](https://arxiv.org/abs/2402.03214)

    这项研究探讨了如何区分人类艺术和AI生成的图像，并提供了几种不同的方法，包括通过监督学习训练的分类器、扩散模型的研究工具以及专业艺术家的知识。这对防止欺诈、遵守政策以及避免模型崩溃都具有重要意义。

    

    生成AI图像的出现完全颠覆了艺术界。从人类艺术中识别AI生成的图像是一个具有挑战性的问题，其影响随着时间的推移而不断增加。未能解决这个问题会导致不良行为者欺诈那些支付高价购买人类艺术品的个人和禁止使用AI图像的公司。这对于需要过滤训练数据以避免潜在模型崩溃的AI模型训练者来说也至关重要。区分人类艺术和AI图像的方法有多种，包括通过监督学习训练的分类器，针对扩散模型的研究工具，以及通过专业艺术家利用他们对艺术技巧的知识进行识别。在本文中，我们试图了解这些方法在现代生成模型的良性和对抗性环境中的表现如何。我们策划了7种风格的真实人类艺术，从5个生成模型生成了与之匹配的图像，并应用了8个检测器。

    The advent of generative AI images has completely disrupted the art world. Identifying AI generated images from human art is a challenging problem whose impact is growing over time. The failure to address this problem allows bad actors to defraud individuals paying a premium for human art, and companies whose stated policies forbid AI imagery. This is also critical for AI model trainers, who need to filter training data to avoid potential model collapse. There are several different approaches to distinguishing human art from AI images, including classifiers trained by supervised learning, research tools targeting diffusion models, and identification by professional artists using their knowledge of artistic techniques. In this paper, we seek to understand how well these approaches can perform against today's modern generative models in both benign and adversarial settings. We curate real human art across 7 styles, generate matching images from 5 generative models, and apply 8 detectors 
    
[^4]: RIDGE: 医学图像分割模型的可重复性、完整性、可靠性、泛化性和效率评估

    RIDGE: Reproducibility, Integrity, Dependability, Generalizability, and Efficiency Assessment of Medical Image Segmentation Models. (arXiv:2401.08847v1 [eess.IV])

    [http://arxiv.org/abs/2401.08847](http://arxiv.org/abs/2401.08847)

    RIDGE是一个用于评估医学图像分割模型的可重复性、完整性、可靠性、泛化性和效率的框架，旨在通过提高工作质量和透明度，确保分割模型在科学可靠性和临床相关性上都具备优势。

    

    深度学习技术尽管具有潜力，但往往缺乏可重复性和泛化性，限制了其在临床中的应用。图像分割是医学图像分析中的关键任务之一，需要对一个或多个感兴趣的区域/体积进行注释。本文介绍了RIDGE清单，这是一个用于评估基于深度学习的医学图像分割模型的可重复性、完整性、可靠性、泛化性和效率的框架。该清单为研究人员提供了指导，以提高其工作的质量和透明度，确保分割模型不仅具有科学的可靠性，还具有临床的相关性。

    Deep learning techniques, despite their potential, often suffer from a lack of reproducibility and generalizability, impeding their clinical adoption. Image segmentation is one of the critical tasks in medical image analysis, in which one or several regions/volumes of interest should be annotated. This paper introduces the RIDGE checklist, a framework for assessing the Reproducibility, Integrity, Dependability, Generalizability, and Efficiency of deep learning-based medical image segmentation models. The checklist serves as a guide for researchers to enhance the quality and transparency of their work, ensuring that segmentation models are not only scientifically sound but also clinically relevant.
    
[^5]: Mixture-of-Experts用于开放域适应的双空间检测方法

    Mixture-of-Experts for Open Set Domain Adaptation: A Dual-Space Detection Approach. (arXiv:2311.00285v1 [cs.CV])

    [http://arxiv.org/abs/2311.00285](http://arxiv.org/abs/2311.00285)

    该论文提出了一种Mixture-of-Experts用于开放域适应的双空间检测方法，利用图像特征空间和路由特征空间之间的不一致性来检测未知类别的样本，无需手动调节阈值。

    

    开放域适应（OSDA）旨在同时处理源域和目标域之间的分布和标签偏移，实现对已知类别的精确分类，同时在目标域中识别未知类别的样本。大多数现有的OSDA方法依赖于深度模型的最终图像特征空间，需要手动调节阈值，并且可能将未知样本错误分类为已知类别。Mixture-of-Expert（MoE）可能是一种解决方法。在MoE中，不同的专家处理不同的输入特征，在路由特征空间中为不同的类别生成独特的专家路由模式。因此，未知类别的样本也可以显示与已知类别不同的专家路由模式。本文提出了双空间检测，利用图像特征空间和路由特征空间之间的不一致性来检测未知类别的样本，无需任何阈值。进一步介绍了图形路由器来更好地利用摘要的信息。

    Open Set Domain Adaptation (OSDA) aims to cope with the distribution and label shifts between the source and target domains simultaneously, performing accurate classification for known classes while identifying unknown class samples in the target domain. Most existing OSDA approaches, depending on the final image feature space of deep models, require manually-tuned thresholds, and may easily misclassify unknown samples as known classes. Mixture-of-Expert (MoE) could be a remedy. Within an MoE, different experts address different input features, producing unique expert routing patterns for different classes in a routing feature space. As a result, unknown class samples may also display different expert routing patterns to known classes. This paper proposes Dual-Space Detection, which exploits the inconsistencies between the image feature space and the routing feature space to detect unknown class samples without any threshold. Graph Router is further introduced to better make use of the
    
[^6]: 改进稠密三维视觉引用的三种方法

    Three Ways to Improve Verbo-visual Fusion for Dense 3D Visual Grounding. (arXiv:2309.04561v1 [cs.CV])

    [http://arxiv.org/abs/2309.04561](http://arxiv.org/abs/2309.04561)

    提出了一个稠密三维引用网络ConcreteNet，包含三个新模块，旨在改善具有相同语义类别干扰因素的重复实例的引用性能。

    

    三维视觉引用是指通过自然语言描述来定位三维场景中被引用的物体的任务。该任务在自主室内机器人到AR/VR等各种应用中广泛应用。目前一种常见的解决方案是通过检测来完成三维视觉引用，即通过边界框来定位。然而，在需要进行物理交互的实际应用中，边界框不足以描述物体的几何属性。因此，我们解决了稠密三维视觉引用的问题，即基于引用的三维实例分割。我们提出了一个稠密三维引用网络ConcreteNet，其中包含三个独立的新模块，旨在改进具有相同语义类别干扰因素的具有挑战性的重复实例的引用性能。首先，我们引入了一个自下而上的注意力融合模块，旨在消除实例间关系线索的歧义性。接下来，我们构造一个cont

    3D visual grounding is the task of localizing the object in a 3D scene which is referred by a description in natural language. With a wide range of applications ranging from autonomous indoor robotics to AR/VR, the task has recently risen in popularity. A common formulation to tackle 3D visual grounding is grounding-by-detection, where localization is done via bounding boxes. However, for real-life applications that require physical interactions, a bounding box insufficiently describes the geometry of an object. We therefore tackle the problem of dense 3D visual grounding, i.e. referral-based 3D instance segmentation. We propose a dense 3D grounding network ConcreteNet, featuring three novel stand-alone modules which aim to improve grounding performance for challenging repetitive instances, i.e. instances with distractors of the same semantic class. First, we introduce a bottom-up attentive fusion module that aims to disambiguate inter-instance relational cues, next we construct a cont
    
[^7]: 通过边际上下文信息的半监督语义分割

    Semi-Supervised Semantic Segmentation via Marginal Contextual Information. (arXiv:2308.13900v1 [cs.CV])

    [http://arxiv.org/abs/2308.13900](http://arxiv.org/abs/2308.13900)

    通过利用分割图中标签的空间相关性，我们提出的S4MC方法在半监督语义分割中通过增强伪标签的方式，并提高了无标签数据的使用量，从而实现了超越现有方法的性能提升。

    

    我们提出了一种新的置信度精化方案，增强了半监督语义分割中的伪标签。与当前主流方法不同的是，我们的方法通过将相邻像素分组并共同考虑它们的伪标签，利用分割图中标签的空间相关性。借助这种上下文信息，我们的方法命名为S4MC，在保持伪标签质量的同时，增加了在训练过程中使用的无标签数据的数量，且计算开销几乎可以忽略不计。通过在标准基准测试上进行大量实验证明，S4MC超越了现有的半监督学习方法，为降低获得稠密标注成本提供了有希望的解决方案。例如，在PASCAL VOC 12上使用366个带注释图像，S4MC比前一最先进方法提高了1.29个mIoU。有关重现我们实验的代码参见...

    We present a novel confidence refinement scheme that enhances pseudo-labels in semi-supervised semantic segmentation. Unlike current leading methods, which filter pixels with low-confidence predictions in isolation, our approach leverages the spatial correlation of labels in segmentation maps by grouping neighboring pixels and considering their pseudo-labels collectively. With this contextual information, our method, named S4MC, increases the amount of unlabeled data used during training while maintaining the quality of the pseudo-labels, all with negligible computational overhead. Through extensive experiments on standard benchmarks, we demonstrate that S4MC outperforms existing state-of-the-art semi-supervised learning approaches, offering a promising solution for reducing the cost of acquiring dense annotations. For example, S4MC achieves a 1.29 mIoU improvement over the prior state-of-the-art method on PASCAL VOC 12 with 366 annotated images. The code to reproduce our experiments i
    
[^8]: 深度神经网络如何学习组合性数据：随机层次模型

    How Deep Neural Networks Learn Compositional Data: The Random Hierarchy Model. (arXiv:2307.02129v1 [cs.LG])

    [http://arxiv.org/abs/2307.02129](http://arxiv.org/abs/2307.02129)

    本文研究了深度神经网络学习组合性数据的问题，通过对随机层次模型进行分类任务，发现深度CNN学习这个任务所需的训练数据数量随着类别数、组合数和迭代次数的增加而渐进增加。

    

    学习一般高维任务是非常困难的，因为它需要与维度成指数增长的训练数据数量。然而，深度卷积神经网络（CNN）在克服这一挑战方面显示出了卓越的成功。一种普遍的假设是可学习任务具有高度结构化，CNN利用这种结构建立了数据的低维表示。然而，我们对它们需要多少训练数据以及这个数字如何取决于数据结构知之甚少。本文回答了针对一个简单的分类任务的这个问题，该任务旨在捕捉真实数据的相关方面：随机层次模型。在这个模型中，$n_c$个类别中的每一个对应于$m$个同义组合的高层次特征，并且这些特征又通过一个重复$L$次的迭代过程由子特征组成。我们发现，需要深度CNN学习这个任务的训练数据数量$P^*$（i）随着$n_c m^L$的增长而渐进地增长，这只有...

    Learning generic high-dimensional tasks is notably hard, as it requires a number of training data exponential in the dimension. Yet, deep convolutional neural networks (CNNs) have shown remarkable success in overcoming this challenge. A popular hypothesis is that learnable tasks are highly structured and that CNNs leverage this structure to build a low-dimensional representation of the data. However, little is known about how much training data they require, and how this number depends on the data structure. This paper answers this question for a simple classification task that seeks to capture relevant aspects of real data: the Random Hierarchy Model. In this model, each of the $n_c$ classes corresponds to $m$ synonymic compositions of high-level features, which are in turn composed of sub-features through an iterative process repeated $L$ times. We find that the number of training data $P^*$ required by deep CNNs to learn this task (i) grows asymptotically as $n_c m^L$, which is only
    
[^9]: 一种从学术论文中提炼文本分类和对象识别的框架

    A Framework For Refining Text Classification and Object Recognition from Academic Articles. (arXiv:2305.17401v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2305.17401](http://arxiv.org/abs/2305.17401)

    本文提出了一种结合基于规则的方法和机器学习的框架，旨在解决从学术论文中提炼文本分类和对象识别的问题。

    

    随着互联网的广泛使用，高效地从大量学术论文中提取特定信息变得越来越重要。数据挖掘技术通常用于解决这个问题。然而，挖掘学术论文的数据具有挑战性，因为它需要自动从复杂的非结构化布局文档中提取特定模式。当前的学术论文数据挖掘方法使用基于规则的（RB）或机器学习（ML）方法。然而，使用基于规则的方法需要编写复杂排版论文的高昂成本。另一方面，仅使用机器学习方法需要对文章中复杂内容类型进行注释工作，这可能成本高昂。此外，仅使用机器学习可能会导致基于规则的方法容易识别的模式被错误提取的情况。为了解决这些问题，本文从分析指定著作中使用的标准布局和排版角度出发，提出了一种结合基于规则的方法和机器学习的框架。

    With the widespread use of the internet, it has become increasingly crucial to extract specific information from vast amounts of academic articles efficiently. Data mining techniques are generally employed to solve this issue. However, data mining for academic articles is challenging since it requires automatically extracting specific patterns in complex and unstructured layout documents. Current data mining methods for academic articles employ rule-based(RB) or machine learning(ML) approaches. However, using rule-based methods incurs a high coding cost for complex typesetting articles. On the other hand, simply using machine learning methods requires annotation work for complex content types within the paper, which can be costly. Furthermore, only using machine learning can lead to cases where patterns easily recognized by rule-based methods are mistakenly extracted. To overcome these issues, from the perspective of analyzing the standard layout and typesetting used in the specified p
    
[^10]: 大规模预训练模型在增量式新类别发现中具有出乎意料的强大表现。

    Large-scale Pre-trained Models are Surprisingly Strong in Incremental Novel Class Discovery. (arXiv:2303.15975v1 [cs.CV])

    [http://arxiv.org/abs/2303.15975](http://arxiv.org/abs/2303.15975)

    本论文提出了一种更加挑战性和实用性的学习方法MSc-iNCD，通过在连续而无人监督的学习中利用大规模预训练模型的丰富先验知识，该方法在增量式新类别发现中表现出出乎意料的强大实力。

    

    在生命长学习者中，从未标记的数据中连续地发现新概念是一个重要的期望。在文献中，这类问题在非常受限的情况下得到了部分解决，其中要么为发现新概念提供有标号的数据（例如 NCD），要么学习在有限数量的增量步骤中发生（例如类 iNCD）。在这项工作中，我们挑战现状，提出了一种更具挑战性和实用性的学习范式，称为 MSc-iNCD，其中学习连续而无人监督，并利用大规模预训练模型的丰富先验知识。为此，我们提出了简单的基线，不仅在较长的学习情境下具有弹性，而且与复杂的最先进方法相比，表现出出乎意料的强大实力。我们在多个基准测试中进行了广泛的实证评估，并展示了我们提出的基线的有效性，大大提升了基准要求。

    Discovering novel concepts from unlabelled data and in a continuous manner is an important desideratum of lifelong learners. In the literature such problems have been partially addressed under very restricted settings, where either access to labelled data is provided for discovering novel concepts (e.g., NCD) or learning occurs for a limited number of incremental steps (e.g., class-iNCD). In this work we challenge the status quo and propose a more challenging and practical learning paradigm called MSc-iNCD, where learning occurs continuously and unsupervisedly, while exploiting the rich priors from large-scale pre-trained models. To this end, we propose simple baselines that are not only resilient under longer learning scenarios, but are surprisingly strong when compared with sophisticated state-of-the-art methods. We conduct extensive empirical evaluation on a multitude of benchmarks and show the effectiveness of our proposed baselines, which significantly raises the bar.
    
[^11]: 深度感知损失网络的系统性能分析：打破迁移学习的约定

    A Systematic Performance Analysis of Deep Perceptual Loss Networks: Breaking Transfer Learning Conventions. (arXiv:2302.04032v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.04032](http://arxiv.org/abs/2302.04032)

    这项工作通过系统评估多种常用的预训练网络及其不同特征提取点，在四个深度感知损失用例上解决了迁移学习中的问题。

    

    深度感知损失是一种在计算机视觉中使用的损失函数，旨在通过使用从神经网络中提取的深度特征来模仿人类感知。近年来，该方法在许多有趣的计算机视觉任务上取得了显著的效果，特别是对于具有图像或类似图像输出的任务，如图像合成、分割、深度预测等。许多应用程序使用预先训练的网络，通常是卷积网络，用于损失计算。尽管对该方法的兴趣和广泛使用增加了，但仍需要更多的努力来探索用于计算深度感知损失的网络以及从哪些层提取特征。本研究旨在通过系统地评估多种常用且易于获取的预训练网络，以及针对四个现有深度感知损失用例的不同特征提取点来纠正这一问题。

    Deep perceptual loss is a type of loss function in computer vision that aims to mimic human perception by using the deep features extracted from neural networks. In recent years, the method has been applied to great effect on a host of interesting computer vision tasks, especially for tasks with image or image-like outputs, such as image synthesis, segmentation, depth prediction, and more. Many applications of the method use pretrained networks, often convolutional networks, for loss calculation. Despite the increased interest and broader use, more effort is needed toward exploring which networks to use for calculating deep perceptual loss and from which layers to extract the features.  This work aims to rectify this by systematically evaluating a host of commonly used and readily available, pretrained networks for a number of different feature extraction points on four existing use cases of deep perceptual loss. The use cases of perceptual similarity, super-resolution, image segmentat
    

