# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Diffusion Models for Adversarial Purification](https://arxiv.org/abs/2403.16067) | 提出一种独立于预训练扩散模型的稳健反向过程，避免了重新训练或微调，有效处理对抗净化中的语义信息损失问题。 |
| [^2] | [Quality-Aware Image-Text Alignment for Real-World Image Quality Assessment](https://arxiv.org/abs/2403.11176) | 提出了一种基于CLIP的自监督方法QualiCLIP，通过质量感知的图像-文本对齐策略，实现了图像质量评估不需要标记MOS的问题 |
| [^3] | [Continual Adversarial Defense](https://arxiv.org/abs/2312.09481) | 提出了第一个能够动态适应任何攻击的持续对抗性防御（CAD）框架。 |
| [^4] | [Efficient Out-of-Distribution Detection with Prototypical Semi-Supervised Learning and Foundation Models](https://arxiv.org/abs/2311.17093) | 本文介绍了一种新的改进的半监督学习方法，利用冻结的基础模型作为神经网络骨干，在半监督学习和超出分布检测方面取得了优越的表现，并引入了新的预训练技术、损失函数和原型选择方法。 |
| [^5] | [A Bayesian Unification of Self-Supervised Clustering and Energy-Based Models.](http://arxiv.org/abs/2401.00873) | 该论文研究了用贝叶斯方法统一自监督聚类和能量模型，提出了一种标准化的推导方法，并设计了一个新的可靠地惩罚失败模式的下界。这个下界使得能够训练一个标准的骨架架构，而无需使用非对称元素。 |
| [^6] | [Mixup Your Own Pairs.](http://arxiv.org/abs/2309.16633) | 本文提出了一种名为SupReMix的方法，通过混合样本，特别是混合负样本和混合正样本，来解决回归问题中表示学习的挑战。这种方法能够提供更好的性能和更准确的回归结果。 |
| [^7] | [PILOT: A Pre-Trained Model-Based Continual Learning Toolbox.](http://arxiv.org/abs/2309.07117) | 本论文介绍了一个名为PILOT的基于预训练模型的持续学习工具箱，为在处理流式数据并适应新数据到来的现实场景中，利用预训练模型进行增量学习提供了一种有前景的方法。 |

# 详细

[^1]: 针对对抗净化的强大扩散模型

    Robust Diffusion Models for Adversarial Purification

    [https://arxiv.org/abs/2403.16067](https://arxiv.org/abs/2403.16067)

    提出一种独立于预训练扩散模型的稳健反向过程，避免了重新训练或微调，有效处理对抗净化中的语义信息损失问题。

    

    基于扩散模型（DM）的对抗净化（AP）已被证明是对抗训练（AT）最有力的替代方法。然而，这些方法忽略了预训练的扩散模型本身对对抗攻击并不稳健这一事实。此外，扩散过程很容易破坏语义信息，在反向过程后生成高质量图像但与原始输入图像完全不同，导致标准精度下降。为了解决这些问题，一个自然的想法是利用对抗训练策略重新训练或微调预训练的扩散模型，然而这在计算上是禁止的。我们提出了一种新颖的具有对抗引导的稳健反向过程，它独立于给定的预训练DMs，并且避免了重新训练或微调DMs。这种强大的引导不仅可以确保生成的净化示例保留更多的语义内容，还可以...

    arXiv:2403.16067v1 Announce Type: cross  Abstract: Diffusion models (DMs) based adversarial purification (AP) has shown to be the most powerful alternative to adversarial training (AT). However, these methods neglect the fact that pre-trained diffusion models themselves are not robust to adversarial attacks as well. Additionally, the diffusion process can easily destroy semantic information and generate a high quality image but totally different from the original input image after the reverse process, leading to degraded standard accuracy. To overcome these issues, a natural idea is to harness adversarial training strategy to retrain or fine-tune the pre-trained diffusion model, which is computationally prohibitive. We propose a novel robust reverse process with adversarial guidance, which is independent of given pre-trained DMs and avoids retraining or fine-tuning the DMs. This robust guidance can not only ensure to generate purified examples retaining more semantic content but also m
    
[^2]: 面向现实世界图像质量评估的质量感知图像-文本对齐

    Quality-Aware Image-Text Alignment for Real-World Image Quality Assessment

    [https://arxiv.org/abs/2403.11176](https://arxiv.org/abs/2403.11176)

    提出了一种基于CLIP的自监督方法QualiCLIP，通过质量感知的图像-文本对齐策略，实现了图像质量评估不需要标记MOS的问题

    

    无参考图像质量评估（NR-IQA）致力于设计一种在没有高质量参考图像的情况下测量图像质量的方法，以符合人类感知，大部分最先进的NR-IQA方法中依赖标注的主观评分（MOS）限制了它们在真实场景中的可扩展性和广泛适用性。为了克服这一限制，我们提出了QualiCLIP（Quality-aware CLIP），这是一种基于CLIP的自监督不需要标记MOS的方法。具体来说，我们引入了一种质量感知的图像-文本对齐策略，使得CLIP生成的表示与图像固有质量相关。从原始图像开始，我们使用不断增加的强度合成地劣化它们。然后，我们训练CLIP根据其与质量相关的反义文本提示的相似性对这些降解图像进行排名，同时保证一致的表达

    arXiv:2403.11176v1 Announce Type: cross  Abstract: No-Reference Image Quality Assessment (NR-IQA) focuses on designing methods to measure image quality in alignment with human perception when a high-quality reference image is unavailable. The reliance on annotated Mean Opinion Scores (MOS) in the majority of state-of-the-art NR-IQA approaches limits their scalability and broader applicability to real-world scenarios. To overcome this limitation, we propose QualiCLIP (Quality-aware CLIP), a CLIP-based self-supervised opinion-unaware method that does not require labeled MOS. In particular, we introduce a quality-aware image-text alignment strategy to make CLIP generate representations that correlate with the inherent quality of the images. Starting from pristine images, we synthetically degrade them with increasing levels of intensity. Then, we train CLIP to rank these degraded images based on their similarity to quality-related antonym text prompts, while guaranteeing consistent represe
    
[^3]: 持续不断的对抗性防御

    Continual Adversarial Defense

    [https://arxiv.org/abs/2312.09481](https://arxiv.org/abs/2312.09481)

    提出了第一个能够动态适应任何攻击的持续对抗性防御（CAD）框架。

    

    针对每月针对视觉分类器的对抗性攻击快速演变的特性，人们提出了许多防御方法，旨在尽可能通用化以抵御尽可能多的已知攻击。然而，设计一个能够对抗所有类型攻击的防御方法并不现实，因为防御系统运行的环境是动态的，包含随着时间出现的各种独特攻击。防御系统必须收集在线少样本对抗反馈以迅速增强自身，充分利用内存。因此，我们提出了第一个能够动态适应任何攻击的持续对抗性防御（CAD）框架，其中各种攻击逐个阶段出现。在实践中，CAD基于四项原则进行建模：(1) 持续适应新攻击而无灾难性遗忘，(2) 少样本适应，(3) 内存高效适应，以及(4) 高准确性

    arXiv:2312.09481v2 Announce Type: replace-cross  Abstract: In response to the rapidly evolving nature of adversarial attacks against visual classifiers on a monthly basis, numerous defenses have been proposed to generalize against as many known attacks as possible. However, designing a defense method that generalizes to all types of attacks is not realistic because the environment in which defense systems operate is dynamic and comprises various unique attacks that emerge as time goes on. The defense system must gather online few-shot defense feedback to promptly enhance itself, leveraging efficient memory utilization. Therefore, we propose the first continual adversarial defense (CAD) framework that adapts to any attacks in a dynamic scenario, where various attacks emerge stage by stage. In practice, CAD is modeled under four principles: (1) continual adaptation to new attacks without catastrophic forgetting, (2) few-shot adaptation, (3) memory-efficient adaptation, and (4) high accur
    
[^4]: 用原型半监督学习和基础模型实现高效的超出分布检测

    Efficient Out-of-Distribution Detection with Prototypical Semi-Supervised Learning and Foundation Models

    [https://arxiv.org/abs/2311.17093](https://arxiv.org/abs/2311.17093)

    本文介绍了一种新的改进的半监督学习方法，利用冻结的基础模型作为神经网络骨干，在半监督学习和超出分布检测方面取得了优越的表现，并引入了新的预训练技术、损失函数和原型选择方法。

    

    本文介绍了PAWS-VMK，一种改进的原型半监督学习方法，专门设计用于利用冻结的基础模型作为神经网络骨干，该方法在计算机视觉领域中优于以往的半监督学习和超出分布（OOD）检测结果，改进了Predicting View-Assignments With Support Samples（PAWS）半监督学习方法。我们引入了(1) 参数化von-Mises Fisher随机邻域嵌入（vMF-SNE）来预训练投影头，使用基础模型的高质量嵌入;(2) 受MixMatch启发的损失，通过对多视图的预测进行平均，提供比PAWS中使用的一致性损失更可靠的监督信号;和(3) 简单k-Means原型选择（SKMPS），一种比其他无监督标签选择方法提供更优越性能的技术。

    arXiv:2311.17093v2 Announce Type: replace-cross  Abstract: This paper describes PAWS-VMK, an improved approach to prototypical semi-supervised learning in the field of computer vision, specifically designed to utilize a frozen foundation model as the neural network backbone. This method outperforms previous results in semi-supervised learning and out-of-distribution (OOD) detection, improving upon the Predicting View-Assignments With Support Samples (PAWS) semi-supervised learning method. We introduce (1) parametric von-Mises Fisher Stochastic Neighbour Embedding (vMF-SNE) to pretrain the projection head using the high-quality embeddings of the foundation model; (2) a MixMatch inspired loss, where predictions across multiple views are averaged to provide a more reliable supervision signal compared to the consistency loss used in PAWS and (3) simple $k$-Means prototype selection (SKMPS), a technique that provides superior performance to other unsupervised label selection approaches in t
    
[^5]: 用贝叶斯方法统一自监督聚类和能量模型

    A Bayesian Unification of Self-Supervised Clustering and Energy-Based Models. (arXiv:2401.00873v1 [cs.LG])

    [http://arxiv.org/abs/2401.00873](http://arxiv.org/abs/2401.00873)

    该论文研究了用贝叶斯方法统一自监督聚类和能量模型，提出了一种标准化的推导方法，并设计了一个新的可靠地惩罚失败模式的下界。这个下界使得能够训练一个标准的骨架架构，而无需使用非对称元素。

    

    自监督学习是一种利用大量无标签数据的流行且强大的方法，文献中提出了各种训练目标。本研究对最先进的自监督学习目标进行贝叶斯分析，阐明了每个类别中潜在的概率图模型，并提出了一种从基本原理出发推导这些模型的标准方法。分析还表明了将自监督学习与基于似然的生成模型自然整合的方法。我们在基于聚类的自监督学习和能量模型领域中实现了这个概念，引入了一个新的下界，经证明能可靠地惩罚最重要的失败模式。此外，这个新提出的下界使得能够训练一个标准的骨干架构，而无需使用诸如停止梯度、动量编码器或专门的聚类等非对称元素。

    Self-supervised learning is a popular and powerful method for utilizing large amounts of unlabeled data, for which a wide variety of training objectives have been proposed in the literature. In this study, we perform a Bayesian analysis of state-of-the-art self-supervised learning objectives, elucidating the underlying probabilistic graphical models in each class and presenting a standardized methodology for their derivation from first principles. The analysis also indicates a natural means of integrating self-supervised learning with likelihood-based generative models. We instantiate this concept within the realm of cluster-based self-supervised learning and energy models, introducing a novel lower bound which is proven to reliably penalize the most important failure modes. Furthermore, this newly proposed lower bound enables the training of a standard backbone architecture without the necessity for asymmetric elements such as stop gradients, momentum encoders, or specialized clusteri
    
[^6]: 混合你自己的对比对

    Mixup Your Own Pairs. (arXiv:2309.16633v1 [cs.LG])

    [http://arxiv.org/abs/2309.16633](http://arxiv.org/abs/2309.16633)

    本文提出了一种名为SupReMix的方法，通过混合样本，特别是混合负样本和混合正样本，来解决回归问题中表示学习的挑战。这种方法能够提供更好的性能和更准确的回归结果。

    

    在表示学习中，回归问题传统上比分类问题受到的关注较少。直接应用为分类设计的表示学习技术到回归问题往往会导致潜空间中碎片化的表示，从而产生次优的性能。本文认为，由于忽视了两个关键方面：序序感知和难度，对于回归问题而言，对比学习的潜能被忽视了。为了解决这些挑战，我们提倡“混合自己的对比对进行监督性对比回归”，而不仅仅依靠真实/增强样本。具体来说，我们提出了混合式监督对比回归学习（SupReMix）。它在嵌入级别上以锚点包含的混合（锚点和一个不同的负样本的混合）作为困难负对，以锚点排除的混合（两个不同的负样本的混合）作为困难正对。这一策略形成了困难样本对学习的方式。

    In representation learning, regression has traditionally received less attention than classification. Directly applying representation learning techniques designed for classification to regression often results in fragmented representations in the latent space, yielding sub-optimal performance. In this paper, we argue that the potential of contrastive learning for regression has been overshadowed due to the neglect of two crucial aspects: ordinality-awareness and hardness. To address these challenges, we advocate "mixup your own contrastive pairs for supervised contrastive regression", instead of relying solely on real/augmented samples. Specifically, we propose Supervised Contrastive Learning for Regression with Mixup (SupReMix). It takes anchor-inclusive mixtures (mixup of the anchor and a distinct negative sample) as hard negative pairs and anchor-exclusive mixtures (mixup of two distinct negative samples) as hard positive pairs at the embedding level. This strategy formulates harde
    
[^7]: PILOT：一个基于预训练模型的持续学习工具箱

    PILOT: A Pre-Trained Model-Based Continual Learning Toolbox. (arXiv:2309.07117v1 [cs.LG])

    [http://arxiv.org/abs/2309.07117](http://arxiv.org/abs/2309.07117)

    本论文介绍了一个名为PILOT的基于预训练模型的持续学习工具箱，为在处理流式数据并适应新数据到来的现实场景中，利用预训练模型进行增量学习提供了一种有前景的方法。

    

    传统机器学习可以有效地解决各种问题，但主要在封闭环境中运作，处理流式数据时存在局限性。作为解决方案，增量学习应运而生，用于处理涉及新数据到来的现实场景。最近，预训练在不断取得重要进展，并引起了众多研究人员的关注。这些预训练模型（PTMs）的强大性能为开发能够有效适应现实场景的持续学习算法提供了有希望的途径。因此，探索在增量学习中利用PTMs已经成为必需。本文介绍了一个名为PILOT的基于预训练模型的持续学习工具箱。一方面，PILOT实施了一些基于预训练模型的最新班级增量学习算法，如L2P、DualPrompt和CODA-Prompt。另一方面，PILOT也适应了典型的班级增量学习场景。

    While traditional machine learning can effectively tackle a wide range of problems, it primarily operates within a closed-world setting, which presents limitations when dealing with streaming data. As a solution, incremental learning emerges to address real-world scenarios involving new data's arrival. Recently, pre-training has made significant advancements and garnered the attention of numerous researchers. The strong performance of these pre-trained models (PTMs) presents a promising avenue for developing continual learning algorithms that can effectively adapt to real-world scenarios. Consequently, exploring the utilization of PTMs in incremental learning has become essential. This paper introduces a pre-trained model-based continual learning toolbox known as PILOT. On the one hand, PILOT implements some state-of-the-art class-incremental learning algorithms based on pre-trained models, such as L2P, DualPrompt, and CODA-Prompt. On the other hand, PILOT also fits typical class-incre
    

