# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fine-tuning with Very Large Dropout](https://arxiv.org/abs/2403.00946) | 通过使用非常高的dropout率进行微调，可以实现超出分布性能，这超出了集成和权重平均方法。 |
| [^2] | [All in One: Exploring Unified Vision-Language Tracking with Multi-Modal Alignment.](http://arxiv.org/abs/2307.03373) | 本文提出了一种一体化视觉-语言跟踪框架，采用统一的Transformer主干网络，实现联合特征提取和交互，提高了在复杂场景下的目标感知能力。 |

# 详细

[^1]: 使用非常大的Dropout进行微调

    Fine-tuning with Very Large Dropout

    [https://arxiv.org/abs/2403.00946](https://arxiv.org/abs/2403.00946)

    通过使用非常高的dropout率进行微调，可以实现超出分布性能，这超出了集成和权重平均方法。

    

    今天不可能假装机器学习实践与训练和测试数据遵循相同分布的观念是兼容的。该论文调查了使用非常高的丢弃率来获得这种丰富表示，尽管使用这样的丢弃率从头开始训练深度网络几乎是不可能的，但在这些条件下对大型预训练模型进行微调不仅是可能的，而且实现了超越集成和权重平均方法的超出分布性能。

    arXiv:2403.00946v1 Announce Type: new  Abstract: It is impossible today to pretend that the practice of machine learning is compatible with the idea that training and testing data follow the same distribution. Several authors have recently used ensemble techniques to show how scenarios involving multiple data distributions are best served by representations that are both richer than those obtained by regularizing for the best in-distribution performance, and richer than those obtained under the influence of the implicit sparsity bias of common stochastic gradient procedures.   This contribution investigates the use of very high dropout rates instead of ensembles to obtain such rich representations. Although training a deep network from scratch using such dropout rates is virtually impossible, fine-tuning a large pre-trained model under such conditions is not only possible but also achieves out-of-distribution performances that exceed those of both ensembles and weight averaging methods
    
[^2]: 一体化视觉-语言跟踪的探索：多模态对齐

    All in One: Exploring Unified Vision-Language Tracking with Multi-Modal Alignment. (arXiv:2307.03373v1 [cs.CV])

    [http://arxiv.org/abs/2307.03373](http://arxiv.org/abs/2307.03373)

    本文提出了一种一体化视觉-语言跟踪框架，采用统一的Transformer主干网络，实现联合特征提取和交互，提高了在复杂场景下的目标感知能力。

    

    当前主流的视觉-语言跟踪框架包括三个部分，即视觉特征提取器、语言特征提取器和融合模型。为了追求更好的性能，视觉-语言跟踪常常使用定制和更重的单模态编码器和多模态融合模型。尽管有效，现有的视觉-语言跟踪器将特征提取和特征集成分开，导致提取的特征缺乏语义引导，在复杂场景下具有有限的目标感知能力，例如相似的干扰物和极端光照。在这项研究中，受到近期在自然语言和计算机视觉任务中统一架构探索的成功启发，我们提出了一种一体化框架，通过采用统一的Transformer主干网络来学习联合特征提取和交互。具体而言，我们混合原始的视觉和语言信号来生成注入语言的视觉单元，然后将它们连接起来。

    Current mainstream vision-language (VL) tracking framework consists of three parts, \ie a visual feature extractor, a language feature extractor, and a fusion model. To pursue better performance, a natural modus operandi for VL tracking is employing customized and heavier unimodal encoders, and multi-modal fusion models. Albeit effective, existing VL trackers separate feature extraction and feature integration, resulting in extracted features that lack semantic guidance and have limited target-aware capability in complex scenarios, \eg similar distractors and extreme illumination. In this work, inspired by the recent success of exploring foundation models with unified architecture for both natural language and computer vision tasks, we propose an All-in-One framework, which learns joint feature extraction and interaction by adopting a unified transformer backbone. Specifically, we mix raw vision and language signals to generate language-injected vision tokens, which we then concatenate
    

