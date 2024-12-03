# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [From Pixels to Insights: A Survey on Automatic Chart Understanding in the Era of Large Foundation Models](https://arxiv.org/abs/2403.12027) | 近年来，随着大型基础模型的兴起，自动图表理解取得了显著进展，本调查论文概述了在这些基础模型背景下图表理解领域的最新发展、挑战和未来方向 |
| [^2] | [Real-time Transformer-based Open-Vocabulary Detection with Efficient Fusion Head](https://arxiv.org/abs/2403.06892) | 本文提出了一种新颖的基于实时变压器的开词汇检测模型OmDet-Turbo，具有高效融合头模块，在实验中取得了与最先进监督模型几乎持平的性能水平。 |
| [^3] | [Exploring Homogeneous and Heterogeneous Consistent Label Associations for Unsupervised Visible-Infrared Person ReID](https://arxiv.org/abs/2402.00672) | 该论文提出了一种同时考虑均质和异质实例级别结构，构建高质量跨模态标签关联的模态统一标签传输方法，用于无监督可见-红外人物重新识别。 |
| [^4] | [Exploring the Influence of Information Entropy Change in Learning Systems.](http://arxiv.org/abs/2309.10625) | 本研究探索了在深度学习系统中引入噪声对性能的影响，证明了特定噪声可以在降低任务复杂性的条件下提升深度架构的性能，通过实验证明了在大规模图像数据集中的显著性能提升。 |
| [^5] | [Cross-Image Context Matters for Bongard Problems.](http://arxiv.org/abs/2309.03468) | Bongard问题是一种需要从一组正负图像中推导出抽象概念并进行分类的智力测试，现有方法在Bongard问题中准确率较低。本研究发现，这是因为现有方法未能整合支持集合中的信息，而是仅依赖于单个支持图像的信息。我们提出了一种通过跨图像上下文来提高准确性的解决方案。 |
| [^6] | [Long-Term Memorability On Advertisements.](http://arxiv.org/abs/2309.00378) | 本研究是首个大规模的记忆性研究，发现广告的长期记忆性对于市场营销非常重要，但在机器学习文献中一直缺乏相关研究。通过分析大量参与者和广告，我们得出了关于什么使广告记忆深刻的有趣见解。 |
| [^7] | [MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities.](http://arxiv.org/abs/2308.02490) | MM-Vet是一个评估标准，用于评估大型多模态模型在复杂任务上的综合能力。该标准解决了如何结构化和评估复杂多模态任务、设计适用于不同问题和回答类型的评估指标以及如何提供模型洞察的问题。通过整合不同的核心视觉-语言能力，MM-Vet展示了有趣的能力和解决复杂任务的方法。 |
| [^8] | [Disentangling the Link Between Image Statistics and Human Perception.](http://arxiv.org/abs/2303.09874) | 本研究直接评估自然图像的概率，并分析它如何影响人类感知。通过展示具有更丰富统计特征的自然图像被感知为具有更大的显着性，论文提供了直接支持Barlow和Attneave理论的证据，并建立了一个新的框架，用于理解图像统计与知觉之间的关系。 |

# 详细

[^1]: 从像素到洞察: 在大型基础模型时代自动图表理解的调查

    From Pixels to Insights: A Survey on Automatic Chart Understanding in the Era of Large Foundation Models

    [https://arxiv.org/abs/2403.12027](https://arxiv.org/abs/2403.12027)

    近年来，随着大型基础模型的兴起，自动图表理解取得了显著进展，本调查论文概述了在这些基础模型背景下图表理解领域的最新发展、挑战和未来方向

    

    数据可视化以图表形式在数据分析中扮演着关键角色，提供关键洞察并帮助做出明智决策。随着近年大型基础模型的崛起，自动图表理解取得了显著进展。基础模型，如大型语言模型(LLMs)，已经在各种自然语言处理（NLP）任务中实现了革命，并越来越多地应用于图表理解任务。本调查论文全面介绍了最新进展、挑战和未来方向，探讨了这些基础模型背景下图表理解的内容。

    arXiv:2403.12027v1 Announce Type: cross  Abstract: Data visualization in the form of charts plays a pivotal role in data analysis, offering critical insights and aiding in informed decision-making. Automatic chart understanding has witnessed significant advancements with the rise of large foundation models in recent years. Foundation models, such as large language models (LLMs), have revolutionized various natural language processing (NLP) tasks and are increasingly being applied to chart understanding tasks. This survey paper provides a comprehensive overview of the recent developments, challenges, and future directions in chart understanding within the context of these foundation models. The paper begins by defining chart understanding, outlining problem formulations, and discussing fundamental building blocks crucial for studying chart understanding tasks. In the section on tasks and datasets, we explore various tasks within chart understanding and discuss their evaluation metrics a
    
[^2]: 基于实时变压器的高效融合头开词汇检测

    Real-time Transformer-based Open-Vocabulary Detection with Efficient Fusion Head

    [https://arxiv.org/abs/2403.06892](https://arxiv.org/abs/2403.06892)

    本文提出了一种新颖的基于实时变压器的开词汇检测模型OmDet-Turbo，具有高效融合头模块，在实验中取得了与最先进监督模型几乎持平的性能水平。

    

    基于端到端变压器的检测器（DETRs）通过整合语言模态，在封闭集和开词汇目标检测（OVD）任务中表现出色，但其高要求的计算需求阻碍了它们在实时目标检测（OD）场景中的实际应用。本文审查了OVDEval基准测试中两个领先模型OmDet和Grounding-DINO的限制，并引入了OmDet-Turbo。这种新颖的基于变压器的实时OVD模型具有创新的高效融合头（EFH）模块，旨在缓解OmDet和Grounding-DINO中观察到的瓶颈。值得注意的是，OmDet-Turbo-Base在应用TensorRT和语言缓存技术后实现了100.2帧每秒（FPS）。值得注意的是，在COCO和LVIS数据集的零样本场景中，OmDet-Turbo的性能几乎与当前最先进的监督模型持平。

    arXiv:2403.06892v1 Announce Type: cross  Abstract: End-to-end transformer-based detectors (DETRs) have shown exceptional performance in both closed-set and open-vocabulary object detection (OVD) tasks through the integration of language modalities. However, their demanding computational requirements have hindered their practical application in real-time object detection (OD) scenarios. In this paper, we scrutinize the limitations of two leading models in the OVDEval benchmark, OmDet and Grounding-DINO, and introduce OmDet-Turbo. This novel transformer-based real-time OVD model features an innovative Efficient Fusion Head (EFH) module designed to alleviate the bottlenecks observed in OmDet and Grounding-DINO. Notably, OmDet-Turbo-Base achieves a 100.2 frames per second (FPS) with TensorRT and language cache techniques applied. Notably, in zero-shot scenarios on COCO and LVIS datasets, OmDet-Turbo achieves performance levels nearly on par with current state-of-the-art supervised models. 
    
[^3]: 探索用于无监督可见-红外人物重新识别的均质和异质一致标签关联

    Exploring Homogeneous and Heterogeneous Consistent Label Associations for Unsupervised Visible-Infrared Person ReID

    [https://arxiv.org/abs/2402.00672](https://arxiv.org/abs/2402.00672)

    该论文提出了一种同时考虑均质和异质实例级别结构，构建高质量跨模态标签关联的模态统一标签传输方法，用于无监督可见-红外人物重新识别。

    

    无监督可见-红外人物重新识别（USL-VI-ReID）旨在无需注释从不同模态中检索相同身份的行人图像。之前的研究侧重于建立跨模态的伪标签关联以弥合模态间的差异，但忽略了在伪标签空间中保持实例级别的均质和异质一致性，导致关联粗糙。为此，我们引入了一个模态统一标签传输（MULT）模块，同时考虑了均质和异质细粒度实例级结构，生成高质量的跨模态标签关联。它建模了均质和异质的关联性，利用它们定义伪标签的不一致性，然后最小化这种不一致性，从而维持了跨模态的对齐并保持了内部模态结构的一致性。此外，还有一个简单易用的在线交叉记忆标签引用模块。

    Unsupervised visible-infrared person re-identification (USL-VI-ReID) aims to retrieve pedestrian images of the same identity from different modalities without annotations. While prior work focuses on establishing cross-modality pseudo-label associations to bridge the modality-gap, they ignore maintaining the instance-level homogeneous and heterogeneous consistency in pseudo-label space, resulting in coarse associations. In response, we introduce a Modality-Unified Label Transfer (MULT) module that simultaneously accounts for both homogeneous and heterogeneous fine-grained instance-level structures, yielding high-quality cross-modality label associations. It models both homogeneous and heterogeneous affinities, leveraging them to define the inconsistency for the pseudo-labels and then minimize it, leading to pseudo-labels that maintain alignment across modalities and consistency within intra-modality structures. Additionally, a straightforward plug-and-play Online Cross-memory Label Ref
    
[^4]: 探索学习系统中信息熵变化的影响

    Exploring the Influence of Information Entropy Change in Learning Systems. (arXiv:2309.10625v1 [cs.AI])

    [http://arxiv.org/abs/2309.10625](http://arxiv.org/abs/2309.10625)

    本研究探索了在深度学习系统中引入噪声对性能的影响，证明了特定噪声可以在降低任务复杂性的条件下提升深度架构的性能，通过实验证明了在大规模图像数据集中的显著性能提升。

    

    在本研究中，我们通过向输入/隐含特征添加噪声来探索深度学习系统中熵变化的影响。本文的应用重点是计算机视觉中的深度学习任务，但所提出的理论可以进一步应用于其他领域。噪声通常被视为各种深度学习架构（如卷积神经网络和视觉变换器）以及图像分类和迁移学习等不同学习任务中的有害扰动。然而，本文旨在重新思考传统命题是否总是成立。我们证明了在特定条件下，特定噪声可以提升各种深度架构的性能。我们在信息熵定义的任务复杂性减少方面从理论上证明了正噪声的增强效果，并在大规模图像数据集（如ImageNet）中实验证明了显著的性能提升。

    In this work, we explore the influence of entropy change in deep learning systems by adding noise to the inputs/latent features. The applications in this paper focus on deep learning tasks within computer vision, but the proposed theory can be further applied to other fields. Noise is conventionally viewed as a harmful perturbation in various deep learning architectures, such as convolutional neural networks (CNNs) and vision transformers (ViTs), as well as different learning tasks like image classification and transfer learning. However, this paper aims to rethink whether the conventional proposition always holds. We demonstrate that specific noise can boost the performance of various deep architectures under certain conditions. We theoretically prove the enhancement gained from positive noise by reducing the task complexity defined by information entropy and experimentally show the significant performance gain in large image datasets, such as the ImageNet. Herein, we use the informat
    
[^5]: 跨图像上下文对于Bongard问题很重要

    Cross-Image Context Matters for Bongard Problems. (arXiv:2309.03468v1 [cs.CV])

    [http://arxiv.org/abs/2309.03468](http://arxiv.org/abs/2309.03468)

    Bongard问题是一种需要从一组正负图像中推导出抽象概念并进行分类的智力测试，现有方法在Bongard问题中准确率较低。本研究发现，这是因为现有方法未能整合支持集合中的信息，而是仅依赖于单个支持图像的信息。我们提出了一种通过跨图像上下文来提高准确性的解决方案。

    

    目前的机器学习方法在解决Bongard问题时存在困难。Bongard问题是一种需要从一组正负“支持”图像中推导出抽象“概念”，然后对于新的查询图像进行分类，判断它是否描述了关键概念的智力测试。在用于自然图像Bongard问题的基准测试Bongard-HOI中，现有方法的准确率仅达到了66%（偶然准确率为50%）。低准确率通常归因于神经网络缺乏发现类似人类符号规则的能力。我们指出，许多现有方法由于一个更简单的问题而失去了准确性：它们没有将支持集合中的信息作为一个整体加入，而是依赖于从单个支持中提取的信息。这是一个关键问题，因为与涉及对象分类的少样本学习任务不同，一个典型的Bongard问题中的“关键概念”只能使用多个正例和多个反例来区分。我们探索了一种解决方案，通过跨图像上下文来提高准确性。

    Current machine learning methods struggle to solve Bongard problems, which are a type of IQ test that requires deriving an abstract "concept" from a set of positive and negative "support" images, and then classifying whether or not a new query image depicts the key concept. On Bongard-HOI, a benchmark for natural-image Bongard problems, existing methods have only reached 66% accuracy (where chance is 50%). Low accuracy is often attributed to neural nets' lack of ability to find human-like symbolic rules. In this work, we point out that many existing methods are forfeiting accuracy due to a much simpler problem: they do not incorporate information contained in the support set as a whole, and rely instead on information extracted from individual supports. This is a critical issue, because unlike in few-shot learning tasks concerning object classification, the "key concept" in a typical Bongard problem can only be distinguished using multiple positives and multiple negatives. We explore a
    
[^6]: 广告的长期记忆性研究

    Long-Term Memorability On Advertisements. (arXiv:2309.00378v1 [cs.CL])

    [http://arxiv.org/abs/2309.00378](http://arxiv.org/abs/2309.00378)

    本研究是首个大规模的记忆性研究，发现广告的长期记忆性对于市场营销非常重要，但在机器学习文献中一直缺乏相关研究。通过分析大量参与者和广告，我们得出了关于什么使广告记忆深刻的有趣见解。

    

    市场营销人员花费数十亿美元在广告上，但是投入到广告上的金钱能起多大作用呢？当顾客在购买时无法辨认出他们看过的品牌的话，花在广告上的钱基本上就被浪费了。尽管在营销中很重要，但迄今为止，在机器学习的文献中还没有关于广告记忆力的研究。大多数研究都是对特定内容类型（如物体和动作视频）进行短期回忆（<5分钟）的研究。另一方面，广告行业只关心长期记忆（几个小时或更长时间），而且广告几乎总是高度多模式化，通过不同的形式（文本、图像和视频）来讲故事。基于这一动机，我们进行了首个大规模记忆性研究，共有1203名参与者和2205个广告涵盖了276个品牌。在不同参与者子群体和广告类型上进行统计测试，我们发现了许多有关什么使广告难忘的有趣见解-无论是内容还是

    Marketers spend billions of dollars on advertisements but to what end? At the purchase time, if customers cannot recognize a brand for which they saw an ad, the money spent on the ad is essentially wasted. Despite its importance in marketing, until now, there has been no study on the memorability of ads in the ML literature. Most studies have been conducted on short-term recall (<5 mins) on specific content types like object and action videos. On the other hand, the advertising industry only cares about long-term memorability (a few hours or longer), and advertisements are almost always highly multimodal, depicting a story through its different modalities (text, images, and videos). With this motivation, we conduct the first large scale memorability study consisting of 1203 participants and 2205 ads covering 276 brands. Running statistical tests over different participant subpopulations and ad-types, we find many interesting insights into what makes an ad memorable - both content and h
    
[^7]: MM-Vet: 评估大型多模态模型的综合能力

    MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities. (arXiv:2308.02490v1 [cs.AI])

    [http://arxiv.org/abs/2308.02490](http://arxiv.org/abs/2308.02490)

    MM-Vet是一个评估标准，用于评估大型多模态模型在复杂任务上的综合能力。该标准解决了如何结构化和评估复杂多模态任务、设计适用于不同问题和回答类型的评估指标以及如何提供模型洞察的问题。通过整合不同的核心视觉-语言能力，MM-Vet展示了有趣的能力和解决复杂任务的方法。

    

    我们提出了MM-Vet，一个评估标准，用于检查在复杂多模态任务上的大型多模态模型（LMM）的表现。最近的LMM展示了各种有趣的能力，例如解决书写在黑板上的数学问题，推理新闻图片中的事件和名人，以及解释视觉笑话。快速的模型进步给评估标准的开发带来了挑战。问题包括：（1）如何系统地构建和评估复杂的多模态任务；（2）如何设计适用于不同类型问题和回答的评估指标；（3）如何给出超出简单性能排名的模型洞察。为此，我们提出了MM-Vet，基于这样一个洞察：解决复杂任务的有趣能力通常通过一种通才模型能够整合不同的核心视觉-语言（VL）能力来实现。MM-Vet定义了6个核心VL能力，并检查了从这些能力组合中得出的16种有趣的整合方式。

    We propose MM-Vet, an evaluation benchmark that examines large multimodal models (LMMs) on complicated multimodal tasks. Recent LMMs have shown various intriguing abilities, such as solving math problems written on the blackboard, reasoning about events and celebrities in news images, and explaining visual jokes. Rapid model advancements pose challenges to evaluation benchmark development. Problems include: (1) How to systematically structure and evaluate the complicated multimodal tasks; (2) How to design evaluation metrics that work well across question and answer types; and (3) How to give model insights beyond a simple performance ranking. To this end, we present MM-Vet, designed based on the insight that the intriguing ability to solve complicated tasks is often achieved by a generalist model being able to integrate different core vision-language (VL) capabilities. MM-Vet defines 6 core VL capabilities and examines the 16 integrations of interest derived from the capability combin
    
[^8]: 图像统计与人类感知之间的关联关系分离

    Disentangling the Link Between Image Statistics and Human Perception. (arXiv:2303.09874v1 [cs.CV])

    [http://arxiv.org/abs/2303.09874](http://arxiv.org/abs/2303.09874)

    本研究直接评估自然图像的概率，并分析它如何影响人类感知。通过展示具有更丰富统计特征的自然图像被感知为具有更大的显着性，论文提供了直接支持Barlow和Attneave理论的证据，并建立了一个新的框架，用于理解图像统计与知觉之间的关系。

    

    在20世纪50年代，霍勒斯巴洛和弗雷德阿特纳夫提出了感官系统和它们如何适应环境之间的关系：早期视觉的进化是为了最大限度地传递关于输入信号的信息。按照香农的定义，这些信息是通过自然场景中拍摄的图像的概率来描述的。由于计算能力的限制，以前无法直接准确地预测图像的概率。尽管这种想法的探索是间接的，主要基于图像密度的过度简化模型或系统设计方法，但这些方法在重现各种生理和心理物理现象方面取得了成功。在本文中，我们直接评估自然图像的概率，并分析它如何确定知觉灵敏度。我们使用与人类意见相关性很高的图像质量指标作为人类视觉的代理，以及一个先进的生成模型来直接估计自然图像的概率密度函数。我们的结果表明，根据Barlow和Attneave理论预测的图像统计与人类知觉之间存在系统性的关联。我们通过展示具有更丰富统计特征的自然图像被感知为具有更大的显着性来说明这一发现，这是通过视觉搜索实验测量的。我们的工作提供了直接支持Barlow和Attneave理论的证据，并建立了一个新的框架，用于理解图像统计与知觉之间的关系。

    In the 1950s Horace Barlow and Fred Attneave suggested a connection between sensory systems and how they are adapted to the environment: early vision evolved to maximise the information it conveys about incoming signals. Following Shannon's definition, this information was described using the probability of the images taken from natural scenes. Previously, direct accurate predictions of image probabilities were not possible due to computational limitations. Despite the exploration of this idea being indirect, mainly based on oversimplified models of the image density or on system design methods, these methods had success in reproducing a wide range of physiological and psychophysical phenomena. In this paper, we directly evaluate the probability of natural images and analyse how it may determine perceptual sensitivity. We employ image quality metrics that correlate well with human opinion as a surrogate of human vision, and an advanced generative model to directly estimate the probabil
    

