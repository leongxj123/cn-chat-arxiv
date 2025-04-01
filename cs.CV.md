# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Masked LoGoNet: Fast and Accurate 3D Image Analysis for Medical Domain](https://arxiv.org/abs/2402.06190) | 本文介绍了一种名为LoGoNet的新型神经网络架构，采用自监督学习方法来应对医学图像分析中的挑战。LoGoNet通过采用大内核注意力和双重编码策略，灵活捕捉长、短距离特征相关性。这种创新的组合技术在医学图像分割中特别有益。 |
| [^2] | [Cultural and Linguistic Diversity Improves Visual Representations.](http://arxiv.org/abs/2310.14356) | 这项研究发现数据集和模型生成的图像描述在不同语言间存在显著的语义差异，多语言数据有更高的语义覆盖率，并且基于多语言训练的模型表现更好。 |
| [^3] | [Interpretable Few-shot Learning with Online Attribute Selection.](http://arxiv.org/abs/2211.09107) | 本文提出了一种在线属性选择机制的天然可解释模型来处理小样本学习，通过减少每个episode中涉及的属性数量提高准确性和可解释性，同时自动检测并补偿人工智能属性池不足的episode。 |

# 详细

[^1]: Masked LoGoNet：用于医学领域的快速准确3D图像分析

    Masked LoGoNet: Fast and Accurate 3D Image Analysis for Medical Domain

    [https://arxiv.org/abs/2402.06190](https://arxiv.org/abs/2402.06190)

    本文介绍了一种名为LoGoNet的新型神经网络架构，采用自监督学习方法来应对医学图像分析中的挑战。LoGoNet通过采用大内核注意力和双重编码策略，灵活捕捉长、短距离特征相关性。这种创新的组合技术在医学图像分割中特别有益。

    

    标准的现代机器学习图像方法在医学应用中面临挑战，因为数据集构建的高成本和有限的标记训练数据。此外，这些方法在部署时通常用于每天处理大量数据，给医疗设施带来高维护成本。在本文中，我们引入了一种新的神经网络架构LoGoNet，采用定制的自监督学习（SSL）方法来缓解这些挑战。LoGoNet在U形架构内整合了一种新颖的特征提取器，利用大内核注意力（LKA）和双重编码策略，灵活地捕捉长、短距离特征相关性。这与现有方法依赖增加网络容量以增强特征提取的方式形成对比。我们模型中这些新技术的组合在医学图像分割中特别有益，考虑到其困难性。

    Standard modern machine-learning-based imaging methods have faced challenges in medical applications due to the high cost of dataset construction and, thereby, the limited labeled training data available. Additionally, upon deployment, these methods are usually used to process a large volume of data on a daily basis, imposing a high maintenance cost on medical facilities. In this paper, we introduce a new neural network architecture, termed LoGoNet, with a tailored self-supervised learning (SSL) method to mitigate such challenges. LoGoNet integrates a novel feature extractor within a U-shaped architecture, leveraging Large Kernel Attention (LKA) and a dual encoding strategy to capture both long-range and short-range feature dependencies adeptly. This is in contrast to existing methods that rely on increasing network capacity to enhance feature extraction. This combination of novel techniques in our model is especially beneficial in medical image segmentation, given the difficulty of le
    
[^2]: 文化和语言多样性提高了视觉表示

    Cultural and Linguistic Diversity Improves Visual Representations. (arXiv:2310.14356v1 [cs.CV] CROSS LISTED)

    [http://arxiv.org/abs/2310.14356](http://arxiv.org/abs/2310.14356)

    这项研究发现数据集和模型生成的图像描述在不同语言间存在显著的语义差异，多语言数据有更高的语义覆盖率，并且基于多语言训练的模型表现更好。

    

    计算机视觉通常将感知视为客观的，并且这种假设在数据集收集和模型训练中得到反映。例如，不同语言的图像描述通常被假定为相同语义内容的翻译。然而，跨文化心理学和语言学的研究表明，个体的视觉感知因其文化背景和所说的语言而异。在本文中，我们展示了在数据集和模型生成的标题中，不同语言之间存在显著的语义内容差异。当数据是多语言而不是单语言时，标题的语义覆盖率平均更高，以场景图、嵌入和语言复杂性进行测量。例如，与一组单语标题相比，多语标题平均有21.8％更多的对象，24.5％更多的关系，以及27.1％更多的属性。此外，使用来自不同语言的内容训练的模型表现最好。

    Computer vision often treats perception as objective, and this assumption gets reflected in the way that datasets are collected and models are trained. For instance, image descriptions in different languages are typically assumed to be translations of the same semantic content. However, work in cross-cultural psychology and linguistics has shown that individuals differ in their visual perception depending on their cultural background and the language they speak. In this paper, we demonstrate significant differences in semantic content across languages in both dataset and model-produced captions. When data is multilingual as opposed to monolingual, captions have higher semantic coverage on average, as measured by scene graph, embedding, and linguistic complexity. For example, multilingual captions have on average 21.8% more objects, 24.5% more relations, and 27.1% more attributes than a set of monolingual captions. Moreover, models trained on content from different languages perform bes
    
[^3]: 在线属性选择的可解释的小样本学习

    Interpretable Few-shot Learning with Online Attribute Selection. (arXiv:2211.09107v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.09107](http://arxiv.org/abs/2211.09107)

    本文提出了一种在线属性选择机制的天然可解释模型来处理小样本学习，通过减少每个episode中涉及的属性数量提高准确性和可解释性，同时自动检测并补偿人工智能属性池不足的episode。

    

    小样本学习(few-shot learning, FSL)是一种挑战性的学习问题，每个类别只有很少的样本可用。在FSL中决策的解释比传统分类更加重要，因为错误的几率更大。然而，大多数以前的FSL方法都是黑匣子模型。本文提出了一种基于易于理解的属性的天然可解释模型来处理FSL。此外，我们提出了一种在线属性选择机制，以有效过滤每个episode中不相关的属性。该属性选择机制通过减少每个episode中涉及的属性数量来提高准确性和可解释性。我们提出了一种机制，自动检测人工智能属性池不足的episode，并通过涉及学习的未知属性来补偿。我们证明了所提出的方法可以实现与黑匣子小样本学习模型相当的结果。

    Few-shot learning (FSL) is a challenging learning problem in which only a few samples are available for each class. Decision interpretation is more important in few-shot classification since there is a greater chance of error than in traditional classification. However, most of the previous FSL methods are black-box models. In this paper, we propose an inherently interpretable model for FSL based on human-friendly attributes. Moreover, we propose an online attribute selection mechanism that can effectively filter out irrelevant attributes in each episode. The attribute selection mechanism improves the accuracy and helps with interpretability by reducing the number of participated attributes in each episode. We propose a mechanism that automatically detects the episodes where the pool of human-friendly attributes are not adequate, and compensates by engaging learned unknown attributes. We demonstrate that the proposed method achieves results on par with black-box few-shot-learning model
    

