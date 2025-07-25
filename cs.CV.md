# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PLOT-TAL -- Prompt Learning with Optimal Transport for Few-Shot Temporal Action Localization](https://arxiv.org/abs/2403.18915) | 提出了使用最优传输进行少样本时序动作定位的提示学习方法，通过多提示学习框架和最优传输理论的结合，有效地捕捉通用特征和减轻过拟合风险 |
| [^2] | [I-CEE: Tailoring Explanations of Image Classification Models to User Expertise.](http://arxiv.org/abs/2312.12102) | I-CEE是一个人为中心的框架，为用户专业知识定制了图像分类模型的解释，通过提供信息丰富的示例图像、局部解释和模型决策来帮助用户理解模型的决策。 |

# 详细

[^1]: 使用最优传输进行少样本时序动作定位的提示学习

    PLOT-TAL -- Prompt Learning with Optimal Transport for Few-Shot Temporal Action Localization

    [https://arxiv.org/abs/2403.18915](https://arxiv.org/abs/2403.18915)

    提出了使用最优传输进行少样本时序动作定位的提示学习方法，通过多提示学习框架和最优传输理论的结合，有效地捕捉通用特征和减轻过拟合风险

    

    本文介绍了一种新的方法来处理少样本学习中的时序动作定位（TAL）。我们的工作解决了传统单样本学习方法的固有局限性，这些方法往往由于无法在真实世界的视频中跨不同上下文进行泛化而导致过拟合。鉴于视频中摄像机视角、背景和物体的多样性，我们提出了一个增强了最优传输的多提示学习框架。这个设计允许模型为每个动作学习一组多样的提示，更有效地捕捉通用特征并分布表示以减轻过拟合的风险。此外，通过采用最优传输理论，我们可以有效地将这些提示与动作特征进行对齐，优化以获得适应视频数据多面性的综合表示。我们的实验证明了动作定位方面的显著改进。

    arXiv:2403.18915v1 Announce Type: cross  Abstract: This paper introduces a novel approach to temporal action localization (TAL) in few-shot learning. Our work addresses the inherent limitations of conventional single-prompt learning methods that often lead to overfitting due to the inability to generalize across varying contexts in real-world videos. Recognizing the diversity of camera views, backgrounds, and objects in videos, we propose a multi-prompt learning framework enhanced with optimal transport. This design allows the model to learn a set of diverse prompts for each action, capturing general characteristics more effectively and distributing the representation to mitigate the risk of overfitting. Furthermore, by employing optimal transport theory, we efficiently align these prompts with action features, optimizing for a comprehensive representation that adapts to the multifaceted nature of video data. Our experiments demonstrate significant improvements in action localization a
    
[^2]: I-CEE: 将图像分类模型的解释定制为用户专业知识

    I-CEE: Tailoring Explanations of Image Classification Models to User Expertise. (arXiv:2312.12102v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2312.12102](http://arxiv.org/abs/2312.12102)

    I-CEE是一个人为中心的框架，为用户专业知识定制了图像分类模型的解释，通过提供信息丰富的示例图像、局部解释和模型决策来帮助用户理解模型的决策。

    

    有效解释黑盒机器学习模型的决策对于依赖它们的人工智能系统的负责任部署至关重要。识别到其重要性，可以生成这些解释的可解释人工智能（XAI）领域提供了几种技术。然而，在这一不断发展的工作中，对用户（解释对象）的关注相对较少，大多数XAI技术产生的是“一刀切”的解释。为了弥合这一差距，实现更加以人为中心的XAI，我们提出了I-CEE，这是一个为用户专业知识定制图像分类解释的框架。受到现有工作的启发，I-CEE通过为用户提供信息丰富的训练数据子集（即示例图像）、相应的局部解释和模型决策来解释图像分类模型的决策。然而，与此前的工作不同的是，I-CEE模拟了示例图像的信息量依赖于用户专业知识的情况，从而为不同的用户提供不同的示例。

    Effectively explaining decisions of black-box machine learning models is critical to responsible deployment of AI systems that rely on them. Recognizing their importance, the field of explainable AI (XAI) provides several techniques to generate these explanations. Yet, there is relatively little emphasis on the user (the explainee) in this growing body of work and most XAI techniques generate "one-size-fits-all" explanations. To bridge this gap and achieve a step closer towards human-centered XAI, we present I-CEE, a framework that provides Image Classification Explanations tailored to User Expertise. Informed by existing work, I-CEE explains the decisions of image classification models by providing the user with an informative subset of training data (i.e., example images), corresponding local explanations, and model decisions. However, unlike prior work, I-CEE models the informativeness of the example images to depend on user expertise, resulting in different examples for different u
    

