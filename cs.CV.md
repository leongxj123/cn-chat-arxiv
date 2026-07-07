# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CausalChaos! Dataset for Comprehensive Causal Action Question Answering Over Longer Causal Chains Grounded in Dynamic Visual Scenes](https://arxiv.org/abs/2404.01299) | 利用卡通图像构建的CausalChaos!数据集，包含更长因果链的因果问答，通过动态互动和视觉展示挑战性因果关系，为模型提供了更多具挑战性且明确定义的因果关系。 |
| [^2] | [Learning to Visually Connect Actions and their Effects.](http://arxiv.org/abs/2401.10805) | 该论文提出了视觉连接动作和其效果的概念（CATE），用于视频理解。研究表明，不同的任务形式产生了捕捉直观动作特性的表示，但模型表现不佳，人类的表现明显优于它们。该研究为未来的努力奠定了基础，并希望能激发出高级形式和模型的灵感。 |

# 详细

[^1]: CausalChaos!数据集：基于动态视觉场景中更长因果链的全面因果行动问答

    CausalChaos! Dataset for Comprehensive Causal Action Question Answering Over Longer Causal Chains Grounded in Dynamic Visual Scenes

    [https://arxiv.org/abs/2404.01299](https://arxiv.org/abs/2404.01299)

    利用卡通图像构建的CausalChaos!数据集，包含更长因果链的因果问答，通过动态互动和视觉展示挑战性因果关系，为模型提供了更多具挑战性且明确定义的因果关系。

    

    因果视频问答（QA）越来越受到关注，然而现有数据集在因果推理分析方面往往缺乏深度。为了填补这一空白，我们利用卡通的独特属性构建了CausalChaos!，这是一个新颖且具有挑战性的因果问答（Why-QA）数据集，基于标志性的“猫和老鼠”卡通系列。我们的数据集通过周到的问题和多层次答案，包含着嵌入动态互动和视觉中的更长因果链，同时动画原理允许动画师创造定义明确、明了的因果关系。这些因素使模型能够解决更具挑战性但明确定义的因果关系。我们还引入了硬负采样，包括CausalConfusion版本。虽然模型表现良好，但仍有很大改进空间，特别是在开放式答案方面。我们确定了更为先进/明确的因果关系建模和联合建模等改进方向。

    arXiv:2404.01299v1 Announce Type: cross  Abstract: Causal video question answering (QA) has garnered increasing interest, yet existing datasets often lack depth in causal reasoning analysis. To address this gap, we capitalize on the unique properties of cartoons and construct CausalChaos!, a novel, challenging causal Why-QA dataset built upon the iconic "Tom and Jerry" cartoon series. With thoughtful questions and multi-level answers, our dataset contains much longer causal chains embedded in dynamic interactions and visuals, at the same time principles of animation allows animators to create well-defined, unambiguous causal relationships. These factors allow models to solve more challenging, yet well-defined causal relationships. We also introduce hard negative mining, including CausalConfusion version. While models perform well, there is much room for improvement, especially, on open-ended answers. We identify more advanced/explicit causal relationship modeling and joint modeling of 
    
[^2]: 学习视觉连接动作和其效果

    Learning to Visually Connect Actions and their Effects. (arXiv:2401.10805v1 [cs.CV])

    [http://arxiv.org/abs/2401.10805](http://arxiv.org/abs/2401.10805)

    该论文提出了视觉连接动作和其效果的概念（CATE），用于视频理解。研究表明，不同的任务形式产生了捕捉直观动作特性的表示，但模型表现不佳，人类的表现明显优于它们。该研究为未来的努力奠定了基础，并希望能激发出高级形式和模型的灵感。

    

    在这项工作中，我们引入了视觉连接动作和其效果（CATE）的新概念，用于视频理解。CATE可以在任务规划和从示范中学习等领域中应用。我们提出了不同基于CATE的任务形式，如动作选择和动作指定，其中视频理解模型以语义和细粒度的方式连接动作和效果。我们观察到不同的形式产生了捕捉直观动作特性的表示。我们还设计了各种基线模型用于动作选择和动作指定。尽管任务具有直观性，但我们观察到模型困难重重，人类表现明显优于它们。本研究旨在为未来的努力奠定基础，展示了连接视频理解中动作和效果的灵活性和多功能性，希望能激发出高级形式和模型的灵感。

    In this work, we introduce the novel concept of visually Connecting Actions and Their Effects (CATE) in video understanding. CATE can have applications in areas like task planning and learning from demonstration. We propose different CATE-based task formulations, such as action selection and action specification, where video understanding models connect actions and effects at semantic and fine-grained levels. We observe that different formulations produce representations capturing intuitive action properties. We also design various baseline models for action selection and action specification. Despite the intuitive nature of the task, we observe that models struggle, and humans outperform them by a large margin. The study aims to establish a foundation for future efforts, showcasing the flexibility and versatility of connecting actions and effects in video understanding, with the hope of inspiring advanced formulations and models.
    

