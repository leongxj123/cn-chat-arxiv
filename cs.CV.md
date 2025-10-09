# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AVicuna: Audio-Visual LLM with Interleaver and Context-Boundary Alignment for Temporal Referential Dialogue](https://arxiv.org/abs/2403.16276) | 介绍了一个新的框架AVicuna，生成了PU-VALOR数据集，解决了音频-视觉时间指代对话中的两个主要挑战：缺乏准确时间注释的数据集和整合复杂时间线索的方法。 |
| [^2] | [Latent Dataset Distillation with Diffusion Models](https://arxiv.org/abs/2403.03881) | 这项研究提出了使用扩散模型进行潜在数据集蒸馏（LD3M），结合潜在空间中的扩散和数据集蒸馏的方法，以解决不同模型架构导致准确性下降和生成高分辨率图像的挑战。 |
| [^3] | [Is my Data in your AI Model? Membership Inference Test with Application to Face Images](https://arxiv.org/abs/2402.09225) | This paper introduces a novel approach called Membership Inference Test (MINT) to empirically assess if specific data was used during the training of AI models. Two MINT architectures based on MLP and CNN are proposed and evaluated on a challenging face recognition task, achieving promising results with up to 90% accuracy. |
| [^4] | [Train-Free Segmentation in MRI with Cubical Persistent Homology.](http://arxiv.org/abs/2401.01160) | 这是一种使用拓扑数据分析进行MRI图像分割的新方法，相比传统机器学习方法具有优势，无需大量注释数据集，提供更可解释和稳定的分割框架。 |
| [^5] | [Attacking the Spike: On the Transferability and Security of Spiking Neural Networks to Adversarial Examples.](http://arxiv.org/abs/2209.03358) | 这项研究主要关注于脉冲神经网络(SNNs)对抗性样本的鲁棒性和转移性。研究发现，成功的白盒对抗攻击SNNs在很大程度上依赖于替代梯度技术，并且非SNN架构创建的对抗样本往往不被SNNs误分类。 |

# 详细

[^1]: AVicuna：具有交错器和上下文边界对齐的音频-视觉LLM用于时间指代对话

    AVicuna: Audio-Visual LLM with Interleaver and Context-Boundary Alignment for Temporal Referential Dialogue

    [https://arxiv.org/abs/2403.16276](https://arxiv.org/abs/2403.16276)

    介绍了一个新的框架AVicuna，生成了PU-VALOR数据集，解决了音频-视觉时间指代对话中的两个主要挑战：缺乏准确时间注释的数据集和整合复杂时间线索的方法。

    

    在日常交流中，人类经常使用语音和手势来指代特定区域或对象，这个过程称为指代对话（RD）。尽管先前的研究已经通过大型语言模型（LLMs）或大型多模型模型（LMMs）在静态环境中调查了RD，但在音频-视觉媒体中探索时间指代对话（TRD）仍然有限。两个主要挑战阻碍了这一领域的进展：（1）缺乏具有精确时间注释的全面未修剪音频-视觉视频数据集，以及（2）需要有效整合复杂的时间听觉和视觉线索的方法。为了解决这些挑战，我们引入了一个新的框架，生成PU-VALOR，这是一个包含超过114,000个未修剪视频的广泛音频-视觉数据集，并介绍了AVicuna，具有音频-视觉令牌交错器（AVTI），确保了时间对齐。

    arXiv:2403.16276v1 Announce Type: cross  Abstract: In everyday communication, humans frequently use speech and gestures to refer to specific areas or objects, a process known as Referential Dialogue (RD). While prior studies have investigated RD through Large Language Models (LLMs) or Large Multimodal Models (LMMs) in static contexts, the exploration of Temporal Referential Dialogue (TRD) within audio-visual media remains limited. Two primary challenges hinder progress in this field: (1) the absence of comprehensive, untrimmed audio-visual video datasets with precise temporal annotations, and (2) the need for methods to integrate complex temporal auditory and visual cues effectively. To address these challenges, we introduce a novel framework to generate PU-VALOR, an extensive audio-visual dataset comprising over 114,000 untrimmed videos with accurate temporal demarcations. We also present AVicuna, featuring an Audio-Visual Tokens Interleaver (AVTI) that ensures the temporal alignment 
    
[^2]: 使用扩散模型进行潜在数据集蒸馏

    Latent Dataset Distillation with Diffusion Models

    [https://arxiv.org/abs/2403.03881](https://arxiv.org/abs/2403.03881)

    这项研究提出了使用扩散模型进行潜在数据集蒸馏（LD3M），结合潜在空间中的扩散和数据集蒸馏的方法，以解决不同模型架构导致准确性下降和生成高分辨率图像的挑战。

    

    机器学习的有效性传统上依赖于越来越大的数据集的可用性。然而，大型数据集带来存储挑战，并且包含一些非影响力样本，在训练过程中可以被忽略而不影响模型最终的准确性。为了应对这些限制，出现了将数据集信息蒸馏成一组压缩样本（合成样本），即蒸馏数据集的概念。其中一个关键方面是选择用于连接原始和合成数据集的架构（通常是ConvNet）。然而，如果所使用的模型架构与蒸馏过程中使用的模型不同，则最终准确性会降低。另一个挑战是生成高分辨率图像，例如128x128及更高。

    arXiv:2403.03881v1 Announce Type: cross  Abstract: The efficacy of machine learning has traditionally relied on the availability of increasingly larger datasets. However, large datasets pose storage challenges and contain non-influential samples, which could be ignored during training without impacting the final accuracy of the model. In response to these limitations, the concept of distilling the information on a dataset into a condensed set of (synthetic) samples, namely a distilled dataset, emerged. One crucial aspect is the selected architecture (usually ConvNet) for linking the original and synthetic datasets. However, the final accuracy is lower if the employed model architecture differs from the model used during distillation. Another challenge is the generation of high-resolution images, e.g., 128x128 and higher. In this paper, we propose Latent Dataset Distillation with Diffusion Models (LD3M) that combine diffusion in latent space with dataset distillation to tackle both chal
    
[^3]: 我的数据在你的AI模型中吗？通过应用于人脸图像的成员推断测试

    Is my Data in your AI Model? Membership Inference Test with Application to Face Images

    [https://arxiv.org/abs/2402.09225](https://arxiv.org/abs/2402.09225)

    This paper introduces a novel approach called Membership Inference Test (MINT) to empirically assess if specific data was used during the training of AI models. Two MINT architectures based on MLP and CNN are proposed and evaluated on a challenging face recognition task, achieving promising results with up to 90% accuracy.

    

    这篇论文介绍了成员推断测试（MINT），一种用于经验性评估特定数据是否被用于训练人工智能（AI）模型的新方法。具体而言，我们提出了两种新颖的MINT架构，旨在学习在经过审计的模型暴露于其训练过程中使用的数据时出现的不同激活模式。第一个架构基于多层感知机（MLP）网络，第二个基于卷积神经网络（CNN）。所提出的MINT架构在具有挑战性的人脸识别任务上进行评估，考虑了三种最先进的人脸识别模型。使用六个公开可用的数据库进行实验，总共包含超过2200万张人脸图像。根据可用的AI模型测试的上下文，考虑了不同的实验场景。有希望的结果达到了90%的准确率。

    arXiv:2402.09225v1 Announce Type: cross Abstract: This paper introduces the Membership Inference Test (MINT), a novel approach that aims to empirically assess if specific data was used during the training of Artificial Intelligence (AI) models. Specifically, we propose two novel MINT architectures designed to learn the distinct activation patterns that emerge when an audited model is exposed to data used during its training process. The first architecture is based on a Multilayer Perceptron (MLP) network and the second one is based on Convolutional Neural Networks (CNNs). The proposed MINT architectures are evaluated on a challenging face recognition task, considering three state-of-the-art face recognition models. Experiments are carried out using six publicly available databases, comprising over 22 million face images in total. Also, different experimental scenarios are considered depending on the context available of the AI model to test. Promising results, up to 90% accuracy, are a
    
[^4]: 无需训练的MRI立方持续同调分割方法

    Train-Free Segmentation in MRI with Cubical Persistent Homology. (arXiv:2401.01160v1 [eess.IV])

    [http://arxiv.org/abs/2401.01160](http://arxiv.org/abs/2401.01160)

    这是一种使用拓扑数据分析进行MRI图像分割的新方法，相比传统机器学习方法具有优势，无需大量注释数据集，提供更可解释和稳定的分割框架。

    

    我们描述了一种新的MRI扫描分割方法，使用拓扑数据分析（TDA），相比传统的机器学习方法具有几个优点。它分为三个步骤，首先通过自动阈值确定要分割的整个对象，然后检测一个已知拓扑结构的独特子集，最后推导出分割的各个组成部分。虽然调用了TDA的经典思想，但这样的算法从未与深度学习方法分离提出。为了实现这一点，我们的方法除了考虑图像的同调性外，还考虑了代表性周期的定位，这是在这种情况下似乎从未被利用过的信息。特别是，它提供了无需大量注释数据集进行分割的能力。TDA还通过将拓扑特征明确映射到分割组件来提供更可解释和稳定的分割框架。

    We describe a new general method for segmentation in MRI scans using Topological Data Analysis (TDA), offering several advantages over traditional machine learning approaches. It works in three steps, first identifying the whole object to segment via automatic thresholding, then detecting a distinctive subset whose topology is known in advance, and finally deducing the various components of the segmentation. Although convoking classical ideas of TDA, such an algorithm has never been proposed separately from deep learning methods. To achieve this, our approach takes into account, in addition to the homology of the image, the localization of representative cycles, a piece of information that seems never to have been exploited in this context. In particular, it offers the ability to perform segmentation without the need for large annotated data sets. TDA also provides a more interpretable and stable framework for segmentation by explicitly mapping topological features to segmentation comp
    
[^5]: 攻击脉冲：关于脉冲神经网络对抗性样本的可转移性与安全性的研究

    Attacking the Spike: On the Transferability and Security of Spiking Neural Networks to Adversarial Examples. (arXiv:2209.03358v3 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2209.03358](http://arxiv.org/abs/2209.03358)

    这项研究主要关注于脉冲神经网络(SNNs)对抗性样本的鲁棒性和转移性。研究发现，成功的白盒对抗攻击SNNs在很大程度上依赖于替代梯度技术，并且非SNN架构创建的对抗样本往往不被SNNs误分类。

    

    脉冲神经网络(SNNs)因其高能效和最近在分类性能上的进展而受到广泛关注。然而，与传统的深度学习方法不同，对SNNs对抗性样本的鲁棒性的分析和研究仍然相对不完善。在这项工作中，我们关注于推进SNNs的对抗攻击方面，并做出了三个主要贡献。首先，我们展示了成功的白盒对抗攻击SNNs在很大程度上依赖于底层的替代梯度技术，即使在对抗性训练SNNs的情况下也一样。其次，利用最佳的替代梯度技术，我们分析了对抗攻击在SNNs和其他最先进的架构如Vision Transformers(ViTs)和Big Transfer Convolutional Neural Networks(CNNs)之间的可转移性。我们证明了非SNN架构创建的对抗样本往往不被SNNs误分类。第三，由于缺乏一个共性

    Spiking neural networks (SNNs) have attracted much attention for their high energy efficiency and for recent advances in their classification performance. However, unlike traditional deep learning approaches, the analysis and study of the robustness of SNNs to adversarial examples remain relatively underdeveloped. In this work, we focus on advancing the adversarial attack side of SNNs and make three major contributions. First, we show that successful white-box adversarial attacks on SNNs are highly dependent on the underlying surrogate gradient technique, even in the case of adversarially trained SNNs. Second, using the best surrogate gradient technique, we analyze the transferability of adversarial attacks on SNNs and other state-of-the-art architectures like Vision Transformers (ViTs) and Big Transfer Convolutional Neural Networks (CNNs). We demonstrate that the adversarial examples created by non-SNN architectures are not misclassified often by SNNs. Third, due to the lack of an ubi
    

