# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Modality Translation for Object Detection Adaptation Without Forgetting Prior Knowledge](https://arxiv.org/abs/2404.01492) | 本文提出了一种ModTr方法，通过小型转换网络调整输入以最小化检测损失，实现了目标检测模型从一个或多个模态到另一个的有效适应，而无需微调参数。 |
| [^2] | [Attention-based Class-Conditioned Alignment for Multi-Source Domain Adaptive Object Detection](https://arxiv.org/abs/2403.09918) | 提出了一种基于注意力的类别条件对齐方案，用于多源领域自适应目标检测，在跨领域对齐每个对象类别的实例。 |
| [^3] | [Demonstrating and Reducing Shortcuts in Vision-Language Representation Learning](https://arxiv.org/abs/2402.17510) | 在视觉-语言表示学习中，论文提出了一种训练和评估框架，引入了合成捷径来探究对比训练是否足以学习到包含所有信息的任务最优表示。 |
| [^4] | [Lite-Mind: Towards Efficient and Robust Brain Representation Network](https://arxiv.org/abs/2312.03781) | Lite-Mind旨在解决fMRI解码中的挑战，通过提出一种高效稳健的脑表示网络，避免了在实践设备上为每个受试者部署特定模型的问题。 |
| [^5] | [Adaptive Self-training Framework for Fine-grained Scene Graph Generation.](http://arxiv.org/abs/2401.09786) | 本论文提出了一种自适应自训练框架用于细粒度场景图生成，通过利用未标注的三元组缓解了场景图生成中的长尾问题。同时，引入了一种新颖的伪标签技术CATM和图结构学习器GSL来提高模型性能。 |
| [^6] | [SC-MIL: Sparsely Coded Multiple Instance Learning for Whole Slide Image Classification.](http://arxiv.org/abs/2311.00048) | 本文提出了SC-MIL模型，通过利用稀疏字典学习来同时改进特征嵌入和实例相关性建模，从而提高全切片图像分类的性能。 |
| [^7] | [Multi-Source Domain Adaptation for Object Detection with Prototype-based Mean-teacher.](http://arxiv.org/abs/2309.14950) | 该论文提出了一种名为Prototype-based Mean-Teacher (PMT)的新型多源域自适应目标检测方法，通过使用类原型而不是域特定子网络来保留域特定信息，提高了准确性和鲁棒性。 |
| [^8] | [Seed Kernel Counting using Domain Randomization and Object Tracking Neural Networks.](http://arxiv.org/abs/2308.05846) | 本研究提出了一种使用领域随机化和目标跟踪神经网络的方法来进行种子核计数。该方法通过使用合成图像作为训练数据的替代品，解决了神经网络模型需要大量有标签训练数据的问题，可以低成本估计谷物产量。 |
| [^9] | [Evaluate Fine-tuning Strategies for Fetal Head Ultrasound Image Segmentation with U-Net.](http://arxiv.org/abs/2307.09067) | 本论文评估了使用U-Net进行胎儿头超声图像分割的微调策略，通过使用轻量级的MobileNet作为编码器，并对有限的图像进行训练，可以获得与从头开始训练相媲美的分割性能，且优于其他策略。 |
| [^10] | [Technical Note: Defining and Quantifying AND-OR Interactions for Faithful and Concise Explanation of DNNs.](http://arxiv.org/abs/2304.13312) | 本文提出了一种通过量化输入变量之间的编码交互来准确且简明地解释深度神经网络(DNN)的推理逻辑的方法。针对此目的，作者提出了两种交互方式，即AND交互和OR交互，并利用它们设计出一系列技术来提高解释的简洁性，同时不会损害准确性。 |
| [^11] | [Ultra-High-Resolution Detector Simulation with Intra-Event Aware GAN and Self-Supervised Relational Reasoning.](http://arxiv.org/abs/2303.08046) | 本文提出了一种新颖的探测器模拟方法IEA-GAN，通过产生与图层相关的上下文化的图像，提高了超高分辨率探测器响应的相关性和多样性。同时，引入新的事件感知损失和统一性损失，显著提高了图像的保真度和多样性。 |

# 详细

[^1]: 不遗忘先验知识的目标检测适应模态转换

    Modality Translation for Object Detection Adaptation Without Forgetting Prior Knowledge

    [https://arxiv.org/abs/2404.01492](https://arxiv.org/abs/2404.01492)

    本文提出了一种ModTr方法，通过小型转换网络调整输入以最小化检测损失，实现了目标检测模型从一个或多个模态到另一个的有效适应，而无需微调参数。

    

    深度学习中常见的做法是在大规模数据集上训练大型神经网络，以在不同领域和任务中准确执行。然而，这种方法在许多应用领域只适用于跨模态，因为使用不同传感器捕获的数据存在更大的分布偏移。本文专注于将大型目标检测模型调整到一个或多个模态的问题，同时保持高效。为此，我们提出了ModTr作为普遍做法微调大型模型的替代方案。ModTr包括使用一个小型转换网络调整输入，该网络经过训练，直接使检测损失最小化。因此，原始模型可以在转换后的输入上工作，无需进行任何进一步的更改或参数微调。对两个知名数据集上从红外到RGB图像的转换的实验结果表明，这种简单的ModTr方法提供了检测器。

    arXiv:2404.01492v1 Announce Type: cross  Abstract: A common practice in deep learning consists of training large neural networks on massive datasets to perform accurately for different domains and tasks. While this methodology may work well in numerous application areas, it only applies across modalities due to a larger distribution shift in data captured using different sensors. This paper focuses on the problem of adapting a large object detection model to one or multiple modalities while being efficient. To do so, we propose ModTr as an alternative to the common approach of fine-tuning large models. ModTr consists of adapting the input with a small transformation network trained to minimize the detection loss directly. The original model can therefore work on the translated inputs without any further change or fine-tuning to its parameters. Experimental results on translating from IR to RGB images on two well-known datasets show that this simple ModTr approach provides detectors tha
    
[^2]: 基于注意力的多源领域自适应目标检测的类别条件对齐

    Attention-based Class-Conditioned Alignment for Multi-Source Domain Adaptive Object Detection

    [https://arxiv.org/abs/2403.09918](https://arxiv.org/abs/2403.09918)

    提出了一种基于注意力的类别条件对齐方案，用于多源领域自适应目标检测，在跨领域对齐每个对象类别的实例。

    

    目标检测（OD）的领域自适应方法致力于通过促进源域和目标域之间的特征对齐来缓解分布转移的影响。多源领域自适应（MSDA）允许利用多个带注释的源数据集和未标记的目标数据来提高检测模型的准确性和鲁棒性。大多数最先进的OD MSDA方法以一种与类别无关的方式执行特征对齐。最近提出的基于原型的方法提出了一种按类别对齐的方法，但由于嘈杂的伪标签而导致错误积累，这可能会对不平衡数据的自适应产生负面影响。为克服这些限制，我们提出了一种基于注意力的类别条件对齐方案，用于MSDA，该方案在跨领域对齐每个对象类别的实例。

    arXiv:2403.09918v1 Announce Type: cross  Abstract: Domain adaptation methods for object detection (OD) strive to mitigate the impact of distribution shifts by promoting feature alignment across source and target domains. Multi-source domain adaptation (MSDA) allows leveraging multiple annotated source datasets, and unlabeled target data to improve the accuracy and robustness of the detection model. Most state-of-the-art MSDA methods for OD perform feature alignment in a class-agnostic manner. This is challenging since the objects have unique modal information due to variations in object appearance across domains. A recent prototype-based approach proposed a class-wise alignment, yet it suffers from error accumulation due to noisy pseudo-labels which can negatively affect adaptation with imbalanced data. To overcome these limitations, we propose an attention-based class-conditioned alignment scheme for MSDA that aligns instances of each object category across domains. In particular, an 
    
[^3]: 示范和减少视觉语言表示学习中的捷径

    Demonstrating and Reducing Shortcuts in Vision-Language Representation Learning

    [https://arxiv.org/abs/2402.17510](https://arxiv.org/abs/2402.17510)

    在视觉-语言表示学习中，论文提出了一种训练和评估框架，引入了合成捷径来探究对比训练是否足以学习到包含所有信息的任务最优表示。

    

    arXiv:2402.17510v1 公告类型: 跨领域 摘要: 视觉-语言模型(VLMs)主要依赖对比训练来学习图像和标题的通用表示。我们关注的情况是当一个图像与多个标题相关联时，每个标题既包含所有标题共享的信息，又包含关于图像场景的每个标题独特的信息。在这种情况下，尚不清楚对比损失是否足以学习包含标题提供的所有信息的任务最优表示，还是对比学习设置是否鼓励学习最小化对比损失的简单捷径。我们引入了视觉-语言的合成捷径：一种训练和评估框架，在其中我们向图像-文本数据注入合成捷径。我们展示了，从头开始训练或用包含这些合成捷径的数据微调的对比VLMs主要学习代表捷径的特征。

    arXiv:2402.17510v1 Announce Type: cross  Abstract: Vision-language models (VLMs) mainly rely on contrastive training to learn general-purpose representations of images and captions. We focus on the situation when one image is associated with several captions, each caption containing both information shared among all captions and unique information per caption about the scene depicted in the image. In such cases, it is unclear whether contrastive losses are sufficient for learning task-optimal representations that contain all the information provided by the captions or whether the contrastive learning setup encourages the learning of a simple shortcut that minimizes contrastive loss. We introduce synthetic shortcuts for vision-language: a training and evaluation framework where we inject synthetic shortcuts into image-text data. We show that contrastive VLMs trained from scratch or fine-tuned with data containing these synthetic shortcuts mainly learn features that represent the shortcu
    
[^4]: Lite-Mind: 高效稳健的脑表示网络

    Lite-Mind: Towards Efficient and Robust Brain Representation Network

    [https://arxiv.org/abs/2312.03781](https://arxiv.org/abs/2312.03781)

    Lite-Mind旨在解决fMRI解码中的挑战，通过提出一种高效稳健的脑表示网络，避免了在实践设备上为每个受试者部署特定模型的问题。

    

    通过非侵入性的fMRI方法解码大脑中的视觉信息的研究正在迅速发展。挑战在于有限的数据可用性和fMRI信号的低信噪比，导致fMRI到图像检索任务的低精度。MindEye技术通过利用高参数计数的深度MLP（每个受试者的996M MLP主干）将fMRI嵌入对齐到CLIP的视觉变换器的最终隐藏层，显着提高了fMRI到图像检索的性能。然而，即使在相同的实验设置内，受试者之间存在显着的个体差异，需要训练特定于受试者的模型。这些大量的参数在实际设备上部署fMRI解码时带来了重大挑战，特别是需要为每个受试者提供特定模型。

    arXiv:2312.03781v2 Announce Type: replace-cross  Abstract: Research in decoding visual information from the brain, particularly through the non-invasive fMRI method, is rapidly progressing. The challenge arises from the limited data availability and the low signal-to-noise ratio of fMRI signals, leading to a low-precision task of fMRI-to-image retrieval. State-of-the-art MindEye remarkably improves fMRI-to-image retrieval performance by leveraging a deep MLP with a high parameter count orders of magnitude, i.e., a 996M MLP Backbone per subject, to align fMRI embeddings to the final hidden layer of CLIP's vision transformer. However, significant individual variations exist among subjects, even within identical experimental setups, mandating the training of subject-specific models. The substantial parameters pose significant challenges in deploying fMRI decoding on practical devices, especially with the necessitating of specific models for each subject. To this end, we propose Lite-Mind,
    
[^5]: 自适应自训练框架用于细粒度场景图生成

    Adaptive Self-training Framework for Fine-grained Scene Graph Generation. (arXiv:2401.09786v1 [cs.CV])

    [http://arxiv.org/abs/2401.09786](http://arxiv.org/abs/2401.09786)

    本论文提出了一种自适应自训练框架用于细粒度场景图生成，通过利用未标注的三元组缓解了场景图生成中的长尾问题。同时，引入了一种新颖的伪标签技术CATM和图结构学习器GSL来提高模型性能。

    

    场景图生成（SGG）模型在基准数据集中存在长尾谓词分布和缺失注释问题。本研究旨在通过利用未标注的三元组缓解SGG的长尾问题。为此，我们引入了一种称为自训练SGG（ST-SGG）的框架，该框架基于未标注的三元组为其分配伪标签以训练SGG模型。虽然在图像识别方面的自训练取得了显著进展，但设计适用于SGG任务的自训练框架更具挑战，因为其固有特性，如语义歧义和长尾分布的谓词类别。因此，我们提出了一种新颖的SGG伪标签技术，称为具有动量的类别自适应阈值化（CATM），它是一种独立于模型的框架，可应用于任何已有的SGG模型。此外，我们设计了一个图结构学习器（GSL），从中获益。

    Scene graph generation (SGG) models have suffered from inherent problems regarding the benchmark datasets such as the long-tailed predicate distribution and missing annotation problems. In this work, we aim to alleviate the long-tailed problem of SGG by utilizing unannotated triplets. To this end, we introduce a Self-Training framework for SGG (ST-SGG) that assigns pseudo-labels for unannotated triplets based on which the SGG models are trained. While there has been significant progress in self-training for image recognition, designing a self-training framework for the SGG task is more challenging due to its inherent nature such as the semantic ambiguity and the long-tailed distribution of predicate classes. Hence, we propose a novel pseudo-labeling technique for SGG, called Class-specific Adaptive Thresholding with Momentum (CATM), which is a model-agnostic framework that can be applied to any existing SGG models. Furthermore, we devise a graph structure learner (GSL) that is benefici
    
[^6]: SC-MIL: 用于全切片图像分类的稀疏编码多实例学习

    SC-MIL: Sparsely Coded Multiple Instance Learning for Whole Slide Image Classification. (arXiv:2311.00048v1 [cs.CV])

    [http://arxiv.org/abs/2311.00048](http://arxiv.org/abs/2311.00048)

    本文提出了SC-MIL模型，通过利用稀疏字典学习来同时改进特征嵌入和实例相关性建模，从而提高全切片图像分类的性能。

    

    多实例学习（MIL）在弱监督的全切片图像（WSI）分类中被广泛使用。典型的MIL方法包括特征嵌入部分，通过预训练的特征提取器将实例嵌入到特征中，以及MIL聚合器，将实例嵌入组合成预测结果。目前的重点是通过自监督预训练来改进这些部分，并单独建模实例之间的相关性。在本文中，我们提出了一种稀疏编码的MIL（SC-MIL），同时通过利用稀疏字典学习来解决这两个方面。稀疏字典学习通过将实例表示为过完备字典中原子的稀疏线性组合来捕捉实例之间的相似性。此外，引入稀疏性可以通过抑制不相关的实例而保留最相关的实例，从而增强实例的特征嵌入。为了改善传统的特征嵌入和实例之间的相关性建模方法，we proposed a sparsely coded MIL.

    Multiple Instance Learning (MIL) has been widely used in weakly supervised whole slide image (WSI) classification. Typical MIL methods include a feature embedding part that embeds the instances into features via a pre-trained feature extractor and the MIL aggregator that combines instance embeddings into predictions. The current focus has been directed toward improving these parts by refining the feature embeddings through self-supervised pre-training and modeling the correlations between instances separately. In this paper, we proposed a sparsely coded MIL (SC-MIL) that addresses those two aspects at the same time by leveraging sparse dictionary learning. The sparse dictionary learning captures the similarities of instances by expressing them as a sparse linear combination of atoms in an over-complete dictionary. In addition, imposing sparsity help enhance the instance feature embeddings by suppressing irrelevant instances while retaining the most relevant ones. To make the convention
    
[^7]: 使用基于原型的均值教师的多源域自适应目标检测

    Multi-Source Domain Adaptation for Object Detection with Prototype-based Mean-teacher. (arXiv:2309.14950v1 [cs.CV])

    [http://arxiv.org/abs/2309.14950](http://arxiv.org/abs/2309.14950)

    该论文提出了一种名为Prototype-based Mean-Teacher (PMT)的新型多源域自适应目标检测方法，通过使用类原型而不是域特定子网络来保留域特定信息，提高了准确性和鲁棒性。

    

    将视觉目标检测器适应于操作目标领域是一项具有挑战性的任务，通常使用无监督域自适应（UDA）方法来实现。当标记的数据集来自多个源域时，将它们视为单独的域并进行多源域自适应（MSDA），相比将这些源域混合并进行UDA，可以提高准确性和鲁棒性，近期的研究也证明了这一点。现有的MSDA方法学习域不变和域特定参数（对于每个源域）来进行自适应。然而，与单源UDA方法不同，学习域特定参数使它们与使用的源域数量成正比增长。本文提出了一种名为基于原型的均值教师（PMT）的新型MSDA方法，该方法使用类原型而不是域特定子网络来保留域特定信息。这些原型是使用对比损失学习的，对齐相同的类别。

    Adapting visual object detectors to operational target domains is a challenging task, commonly achieved using unsupervised domain adaptation (UDA) methods. When the labeled dataset is coming from multiple source domains, treating them as separate domains and performing a multi-source domain adaptation (MSDA) improves the accuracy and robustness over mixing these source domains and performing a UDA, as observed by recent studies in MSDA. Existing MSDA methods learn domain invariant and domain-specific parameters (for each source domain) for the adaptation. However, unlike single-source UDA methods, learning domain-specific parameters makes them grow significantly proportional to the number of source domains used. This paper proposes a novel MSDA method called Prototype-based Mean-Teacher (PMT), which uses class prototypes instead of domain-specific subnets to preserve domain-specific information. These prototypes are learned using a contrastive loss, aligning the same categories across 
    
[^8]: 使用领域随机化和目标跟踪神经网络进行种子核计数

    Seed Kernel Counting using Domain Randomization and Object Tracking Neural Networks. (arXiv:2308.05846v1 [cs.CV])

    [http://arxiv.org/abs/2308.05846](http://arxiv.org/abs/2308.05846)

    本研究提出了一种使用领域随机化和目标跟踪神经网络的方法来进行种子核计数。该方法通过使用合成图像作为训练数据的替代品，解决了神经网络模型需要大量有标签训练数据的问题，可以低成本估计谷物产量。

    

    高通量表型（HTP）对种子的评估是对生长、发育、耐受性、抗性、生态、产量等复杂种子特性的全面评估，以及衡量形成更复杂特性的参数。种子表型的关键之一是谷物产量估计，种子生产行业依赖于这一估计来进行业务运作。目前市场上已有机械化的种子核计数器，但价格往往很高，有时超出小规模种子生产企业的承受范围。目标跟踪神经网络模型(如YOLO)的发展使计算机科学家能够设计出可以低成本估计谷物产量的算法。神经网络模型的关键瓶颈是需要大量有标签的训练数据才能投入使用。我们证明了使用合成图像作为可行替代方案。

    High-throughput phenotyping (HTP) of seeds, also known as seed phenotyping, is the comprehensive assessment of complex seed traits such as growth, development, tolerance, resistance, ecology, yield, and the measurement of parameters that form more complex traits. One of the key aspects of seed phenotyping is cereal yield estimation that the seed production industry relies upon to conduct their business. While mechanized seed kernel counters are available in the market currently, they are often priced high and sometimes outside the range of small scale seed production firms' affordability. The development of object tracking neural network models such as You Only Look Once (YOLO) enables computer scientists to design algorithms that can estimate cereal yield inexpensively. The key bottleneck with neural network models is that they require a plethora of labelled training data before they can be put to task. We demonstrate that the use of synthetic imagery serves as a feasible substitute t
    
[^9]: 评估使用U-Net进行胎儿头超声图像分割中的微调策略

    Evaluate Fine-tuning Strategies for Fetal Head Ultrasound Image Segmentation with U-Net. (arXiv:2307.09067v1 [eess.IV])

    [http://arxiv.org/abs/2307.09067](http://arxiv.org/abs/2307.09067)

    本论文评估了使用U-Net进行胎儿头超声图像分割的微调策略，通过使用轻量级的MobileNet作为编码器，并对有限的图像进行训练，可以获得与从头开始训练相媲美的分割性能，且优于其他策略。

    

    胎儿头分割是测量妊娠期间胎儿头围(HC)的关键步骤，是监测胎儿生长的重要生物测定学。然而，手动生成生物学测定是耗时且结果不一致的。为解决这个问题，我们提出了一种迁移学习（TL）方法，通过细调(U-Net网络和轻量级的MobileNet作为编码器)对一组有限的胎儿头超声图像进行分割。这种方法解决了从头开始训练CNN网络的挑战。研究表明，我们提出的细调策略在训练参数减少85.8%的情况下，能够获得可比较的分割性能。并且，我们的细调策略优于其他策略。

    Fetal head segmentation is a crucial step in measuring the fetal head circumference (HC) during gestation, an important biometric in obstetrics for monitoring fetal growth. However, manual biometry generation is time-consuming and results in inconsistent accuracy. To address this issue, convolutional neural network (CNN) models have been utilized to improve the efficiency of medical biometry. But training a CNN network from scratch is a challenging task, we proposed a Transfer Learning (TL) method. Our approach involves fine-tuning (FT) a U-Net network with a lightweight MobileNet as the encoder to perform segmentation on a set of fetal head ultrasound (US) images with limited effort. This method addresses the challenges associated with training a CNN network from scratch. It suggests that our proposed FT strategy yields segmentation performance that is comparable when trained with a reduced number of parameters by 85.8%. And our proposed FT strategy outperforms other strategies with s
    
[^10]: 技术笔记：定义和量化DNN的AND-OR交互以进行准确和简明的解释

    Technical Note: Defining and Quantifying AND-OR Interactions for Faithful and Concise Explanation of DNNs. (arXiv:2304.13312v1 [cs.LG])

    [http://arxiv.org/abs/2304.13312](http://arxiv.org/abs/2304.13312)

    本文提出了一种通过量化输入变量之间的编码交互来准确且简明地解释深度神经网络(DNN)的推理逻辑的方法。针对此目的，作者提出了两种交互方式，即AND交互和OR交互，并利用它们设计出一系列技术来提高解释的简洁性，同时不会损害准确性。

    

    本文旨在通过量化输入变量之间的编码交互来解释深度神经网络(DNN)的推理逻辑。具体而言，我们首先重新思考交互的定义，然后正式定义了基于交互的解释的准确性和简洁性。为此，我们提出了两种交互方式，即AND交互和OR交互。针对准确性，我们证明了AND（OR）交互在量化输入变量之间的AND（OR）关系效应方面的唯一性。此外，基于AND-OR交互，我们设计了技术来提高解释的简洁性，同时不会损害准确性。因此，DNN的推理逻辑可以通过一组符号概念准确而简明地解释。

    In this technical note, we aim to explain a deep neural network (DNN) by quantifying the encoded interactions between input variables, which reflects the DNN's inference logic. Specifically, we first rethink the definition of interactions, and then formally define faithfulness and conciseness for interaction-based explanation. To this end, we propose two kinds of interactions, i.e., the AND interaction and the OR interaction. For faithfulness, we prove the uniqueness of the AND (OR) interaction in quantifying the effect of the AND (OR) relationship between input variables. Besides, based on AND-OR interactions, we design techniques to boost the conciseness of the explanation, while not hurting the faithfulness. In this way, the inference logic of a DNN can be faithfully and concisely explained by a set of symbolic concepts.
    
[^11]: 基于事件感知的生成对抗网络和自监督关系推理的超高分辨率探测器模拟

    Ultra-High-Resolution Detector Simulation with Intra-Event Aware GAN and Self-Supervised Relational Reasoning. (arXiv:2303.08046v1 [physics.ins-det])

    [http://arxiv.org/abs/2303.08046](http://arxiv.org/abs/2303.08046)

    本文提出了一种新颖的探测器模拟方法IEA-GAN，通过产生与图层相关的上下文化的图像，提高了超高分辨率探测器响应的相关性和多样性。同时，引入新的事件感知损失和统一性损失，显著提高了图像的保真度和多样性。

    

    在粒子物理学中，模拟高分辨率探测器响应一直是一个存储成本高、计算密集的过程。尽管深度生成模型可以使这个过程更具成本效益，但超高分辨率探测器模拟仍然很困难，因为它包含了事件内相关和细粒度的相互信息。为了克服这些限制，我们提出了一种新颖的生成对抗网络方法（IEA-GAN），融合了自监督学习和关系推理模型。IEA-GAN提出了一个关系推理模块，近似于探测器模拟中“事件”的概念，可以生成与图层相关的上下文化的图像，提高了超高分辨率探测器响应的相关性和多样性。IEA-GAN还引入了新的事件感知损失和统一性损失，显著提高了图像的保真度和多样性。我们展示了IEA-GAN的应用。

    Simulating high-resolution detector responses is a storage-costly and computationally intensive process that has long been challenging in particle physics. Despite the ability of deep generative models to make this process more cost-efficient, ultra-high-resolution detector simulation still proves to be difficult as it contains correlated and fine-grained mutual information within an event. To overcome these limitations, we propose Intra-Event Aware GAN (IEA-GAN), a novel fusion of Self-Supervised Learning and Generative Adversarial Networks. IEA-GAN presents a Relational Reasoning Module that approximates the concept of an ''event'' in detector simulation, allowing for the generation of correlated layer-dependent contextualized images for high-resolution detector responses with a proper relational inductive bias. IEA-GAN also introduces a new intra-event aware loss and a Uniformity loss, resulting in significant enhancements to image fidelity and diversity. We demonstrate IEA-GAN's ap
    

