# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improving Medical Multi-modal Contrastive Learning with Expert Annotations](https://arxiv.org/abs/2403.10153) | eCLIP是一种改进的CLIP模型，通过集成专家注释和混合增强来应对医学影像分析中的数据稀缺和模态差距挑战，提高了模型学习效果 |
| [^2] | [GET: Unlocking the Multi-modal Potential of CLIP for Generalized Category Discovery](https://arxiv.org/abs/2403.09974) | 本文提出了一种文本嵌入合成器（TES），用于为无标签数据生成伪文本嵌入，以解锁CLIP用于广义类别发现任务中的多模态潜力。 |
| [^3] | [Probabilistic Routing for Graph-Based Approximate Nearest Neighbor Search](https://arxiv.org/abs/2402.11354) | 该论文提出了一种基于概率路由的方法，通过引入PEOs有效识别图中需要考虑进行精确距离计算的邻居，从而显著提高了基于图的近似最近邻搜索的效率。 |
| [^4] | [Manipulating Feature Visualizations with Gradient Slingshots.](http://arxiv.org/abs/2401.06122) | 本研究探究了激活最大化方法在对抗模型操作中的脆弱性，并提出了一种新的方法来操纵特征可视化，以隐藏特定神经元的功能。 |
| [^5] | [Complementary Information Mutual Learning for Multimodality Medical Image Segmentation.](http://arxiv.org/abs/2401.02717) | 这篇论文介绍了一种称为互补信息相互学习（CIML）的框架，在多模态医学影像分割中解决了模态间冗余信息的负面影响。通过采用加法和任务分解的方法，CIML成功地消除了冗余信息，提高了分割的准确性。 |

# 详细

[^1]: 改进医学多模态对比学习与专家注释

    Improving Medical Multi-modal Contrastive Learning with Expert Annotations

    [https://arxiv.org/abs/2403.10153](https://arxiv.org/abs/2403.10153)

    eCLIP是一种改进的CLIP模型，通过集成专家注释和混合增强来应对医学影像分析中的数据稀缺和模态差距挑战，提高了模型学习效果

    

    我们介绍了一种增强版CLIP模型——eCLIP，它集成了放射科医生眼球注视热图形式的专家注释。它解决了对比多模态医学影像分析中的关键挑战，尤其是数据稀缺和“模态差距”——图像和文本嵌入之间存在的显著差异，降低了表示的质量并阻碍了跨模态互操作性。eCLIP集成了一个热图处理器，并利用混合增强来有效利用稀缺的专家注释，从而提高模型的学习效果。eCLIP设计为通用的，适用于任何形式的CLIP变体，无需修改核心架构。通过对多个任务的详细评估，包括零样本推断、线性探针、跨模态检索以及使用冻结的大型语言模型进行放射学报告的检索增强生成（RAG），eCLIP展示了其...

    arXiv:2403.10153v1 Announce Type: cross  Abstract: We introduce eCLIP, an enhanced version of the CLIP model that integrates expert annotations in the form of radiologist eye-gaze heatmaps. It tackles key challenges in contrastive multi-modal medical imaging analysis, notably data scarcity and the "modality gap" -- a significant disparity between image and text embeddings that diminishes the quality of representations and hampers cross-modal interoperability. eCLIP integrates a heatmap processor and leverages mixup augmentation to efficiently utilize the scarce expert annotations, thus boosting the model's learning effectiveness. eCLIP is designed to be generally applicable to any variant of CLIP without requiring any modifications of the core architecture. Through detailed evaluations across several tasks, including zero-shot inference, linear probing, cross-modal retrieval, and Retrieval Augmented Generation (RAG) of radiology reports using a frozen Large Language Model, eCLIP showca
    
[^2]: GET：解锁CLIP的多模态潜力，用于广义类别发现

    GET: Unlocking the Multi-modal Potential of CLIP for Generalized Category Discovery

    [https://arxiv.org/abs/2403.09974](https://arxiv.org/abs/2403.09974)

    本文提出了一种文本嵌入合成器（TES），用于为无标签数据生成伪文本嵌入，以解锁CLIP用于广义类别发现任务中的多模态潜力。

    

    给定包含旧类别和新类别的无标签数据集，广义类别发现（GCD）旨在准确发现新类别，并正确分类旧类别，利用从有标签样本中学习的类别概念。当前的GCD方法只使用单一的视觉信息模态，导致在视觉上相似类别的分类效果不佳。虽然某些类别在视觉上容易混淆，但它们的文本信息可能是不同的，这促使我们将文本信息引入到GCD任务中。然而，无标签数据缺乏类别名称，使得利用文本信息变得不切实际。为了解决这一具有挑战性的问题，在本文中，我们提出了一种文本嵌入合成器（TES），用于为无标签样本生成伪文本嵌入。具体而言，我们的TES利用CLIP可以生成对齐的视觉-语言特征这一特性，将视觉嵌入转换为CLIP文本模型的标记。

    arXiv:2403.09974v1 Announce Type: cross  Abstract: Given unlabelled datasets containing both old and new categories, generalized category discovery (GCD) aims to accurately discover new classes while correctly classifying old classes, leveraging the class concepts learned from labeled samples. Current GCD methods only use a single visual modality of information, resulting in poor classification of visually similar classes. Though certain classes are visually confused, their text information might be distinct, motivating us to introduce text information into the GCD task. However, the lack of class names for unlabelled data makes it impractical to utilize text information. To tackle this challenging problem, in this paper, we propose a Text Embedding Synthesizer (TES) to generate pseudo text embeddings for unlabelled samples. Specifically, our TES leverages the property that CLIP can generate aligned vision-language features, converting visual embeddings into tokens of the CLIP's text e
    
[^3]: 基于概率路由的基于图的近似最近邻搜索

    Probabilistic Routing for Graph-Based Approximate Nearest Neighbor Search

    [https://arxiv.org/abs/2402.11354](https://arxiv.org/abs/2402.11354)

    该论文提出了一种基于概率路由的方法，通过引入PEOs有效识别图中需要考虑进行精确距离计算的邻居，从而显著提高了基于图的近似最近邻搜索的效率。

    

    arXiv：2402.11354v1 公告类型：交叉 摘要：在机器学习领域，高维空间中的近似最近邻搜索(ANNS)是一个重要挑战。近年来，基于图的方法已经成为ANNS的优越方法，建立了一种新的技术水平。尽管引入了各种基于图的ANNS优化方法，但它们主要依赖于缺乏正式理论支持的启发式方法。本文旨在通过引入一种方法来增强基于图的ANNS中的路由，该方法在探索图中节点的邻居时提供概率保证。我们将问题建模为概率路由，并通过结合局部敏感技术开发了两种基准策略。随后，我们介绍了PEOs，这是一种有效识别图中应考虑进行精确距离计算的邻居的新方法，从而在实践中显著提高了效率。我们的实验证明...

    arXiv:2402.11354v1 Announce Type: cross  Abstract: Approximate nearest neighbor search (ANNS) in high-dimensional spaces is a pivotal challenge in the field of machine learning. In recent years, graph-based methods have emerged as the superior approach to ANNS, establishing a new state of the art. Although various optimizations for graph-based ANNS have been introduced, they predominantly rely on heuristic methods that lack formal theoretical backing. This paper aims to enhance routing within graph-based ANNS by introducing a method that offers a probabilistic guarantee when exploring a node's neighbors in the graph. We formulate the problem as probabilistic routing and develop two baseline strategies by incorporating locality-sensitive techniques. Subsequently, we introduce PEOs, a novel approach that efficiently identifies which neighbors in the graph should be considered for exact distance computation, thus significantly improving efficiency in practice. Our experiments demonstrate 
    
[^4]: 用梯度弹射操纵特征可视化

    Manipulating Feature Visualizations with Gradient Slingshots. (arXiv:2401.06122v1 [cs.LG])

    [http://arxiv.org/abs/2401.06122](http://arxiv.org/abs/2401.06122)

    本研究探究了激活最大化方法在对抗模型操作中的脆弱性，并提出了一种新的方法来操纵特征可视化，以隐藏特定神经元的功能。

    

    深度神经网络(DNNs)能够学习复杂而多样化的表示，然而，学习到的概念的语义性质仍然未知。解释DNNs学习到的概念的常用方法是激活最大化(AM)，它生成一个合成的输入信号，最大化激活网络中的特定神经元。在本文中，我们研究了这种方法对于对抗模型操作的脆弱性，并引入了一种新的方法来操纵特征可视化，而不改变模型结构或对模型的决策过程产生显著影响。我们评估了我们的方法对几个神经网络模型的效果，并展示了它隐藏特定神经元功能的能力，在模型审核过程中使用选择的目标解释屏蔽了原始解释。作为一种补救措施，我们提出了一种防止这种操纵的防护措施，并提供了定量证据，证明了它的有效性。

    Deep Neural Networks (DNNs) are capable of learning complex and versatile representations, however, the semantic nature of the learned concepts remains unknown. A common method used to explain the concepts learned by DNNs is Activation Maximization (AM), which generates a synthetic input signal that maximally activates a particular neuron in the network. In this paper, we investigate the vulnerability of this approach to adversarial model manipulations and introduce a novel method for manipulating feature visualization without altering the model architecture or significantly impacting the model's decision-making process. We evaluate the effectiveness of our method on several neural network models and demonstrate its capabilities to hide the functionality of specific neurons by masking the original explanations of neurons with chosen target explanations during model auditing. As a remedy, we propose a protective measure against such manipulations and provide quantitative evidence which 
    
[^5]: 多模态医学影像分割的互补信息相互学习

    Complementary Information Mutual Learning for Multimodality Medical Image Segmentation. (arXiv:2401.02717v1 [cs.CV])

    [http://arxiv.org/abs/2401.02717](http://arxiv.org/abs/2401.02717)

    这篇论文介绍了一种称为互补信息相互学习（CIML）的框架，在多模态医学影像分割中解决了模态间冗余信息的负面影响。通过采用加法和任务分解的方法，CIML成功地消除了冗余信息，提高了分割的准确性。

    

    由于医学影像的局限性和肿瘤信号的多样性，放射科医生必须利用多模态图像进行肿瘤分割和诊断。这导致了多模态学习在分割中的发展。然而，模态之间的冗余性给现有的基于减法的联合学习方法带来了挑战，例如错误判断模态的重要性，忽视特定的模态信息，以及增加认知负荷。这些棘手的问题最终降低了分割的准确性并增加了过拟合的风险。本文提出了互补信息相互学习（CIML）框架，可以对模态间冗余信息的负面影响进行数学建模和解决。CIML采用了加法的思想，并通过归纳偏置驱动的任务分解和基于消息传递的冗余性过滤来消除模态间的冗余信息。CIML将多模态分割任务首先分解为多个子任务

    Radiologists must utilize multiple modal images for tumor segmentation and diagnosis due to the limitations of medical imaging and the diversity of tumor signals. This leads to the development of multimodal learning in segmentation. However, the redundancy among modalities creates challenges for existing subtraction-based joint learning methods, such as misjudging the importance of modalities, ignoring specific modal information, and increasing cognitive load. These thorny issues ultimately decrease segmentation accuracy and increase the risk of overfitting. This paper presents the complementary information mutual learning (CIML) framework, which can mathematically model and address the negative impact of inter-modal redundant information. CIML adopts the idea of addition and removes inter-modal redundant information through inductive bias-driven task decomposition and message passing-based redundancy filtering. CIML first decomposes the multimodal segmentation task into multiple subta
    

