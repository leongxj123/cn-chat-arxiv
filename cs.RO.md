# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Behavior Generation with Latent Actions](https://arxiv.org/abs/2403.03181) | 这项工作介绍了一种名为矢量量化行为转换器（VQ-BeT）的多功能行为生成模型，通过对连续动作进行标记化处理多模态动作预测、条件生成和部分观察。 |
| [^2] | [Dynamic planning in hierarchical active inference](https://arxiv.org/abs/2402.11658) | 通过研究在动态规划领域中模拟工具使用的目标，我们深入探讨了主动推断中的动态规划，该领域考虑到生物目标导向行为的两个关键方面 |
| [^3] | [Exploring the Dynamics between Cobot's Production Rhythm, Locus of Control and Emotional State in a Collaborative Assembly Scenario](https://arxiv.org/abs/2402.00808) | 本研究探索了在协作装配场景中，Cobot的生产节奏对参与者的经验性定位控制和情绪状态的影响。虽然在情绪状态和定位控制方面未发现差异，但考虑到其他心理变量，结果表明需要考虑个体的心理特征来提供更好的互动体验。 |

# 详细

[^1]: 具有潜在动作的行为生成

    Behavior Generation with Latent Actions

    [https://arxiv.org/abs/2403.03181](https://arxiv.org/abs/2403.03181)

    这项工作介绍了一种名为矢量量化行为转换器（VQ-BeT）的多功能行为生成模型，通过对连续动作进行标记化处理多模态动作预测、条件生成和部分观察。

    

    从带标签的数据集中生成复杂行为的生成建模一直是决策制定中长期存在的问题。与语言或图像生成不同，决策制定需要建模动作 - 连续值向量，其在分布上是多模态的，可能来自未经筛选的来源，在顺序预测中生成误差可能会相互累积。最近一类称为行为转换器（BeT）的模型通过使用k-means聚类对动作进行离散化以捕捉不同模式来解决这个问题。然而，k-means在处理高维动作空间或长序列时存在困难，并且缺乏梯度信息，因此BeT在建模长距离动作时存在困难。在这项工作中，我们提出了一种名为矢量量化行为转换器（VQ-BeT）的多功能行为生成模型，用于处理多模态动作预测、条件生成和部分观察。VQ-BeT通过对连续动作进行标记化来增强BeT

    arXiv:2403.03181v1 Announce Type: cross  Abstract: Generative modeling of complex behaviors from labeled datasets has been a longstanding problem in decision making. Unlike language or image generation, decision making requires modeling actions - continuous-valued vectors that are multimodal in their distribution, potentially drawn from uncurated sources, where generation errors can compound in sequential prediction. A recent class of models called Behavior Transformers (BeT) addresses this by discretizing actions using k-means clustering to capture different modes. However, k-means struggles to scale for high-dimensional action spaces or long sequences, and lacks gradient information, and thus BeT suffers in modeling long-range actions. In this work, we present Vector-Quantized Behavior Transformer (VQ-BeT), a versatile model for behavior generation that handles multimodal action prediction, conditional generation, and partial observations. VQ-BeT augments BeT by tokenizing continuous
    
[^2]: 分层主动推断中的动态规划

    Dynamic planning in hierarchical active inference

    [https://arxiv.org/abs/2402.11658](https://arxiv.org/abs/2402.11658)

    通过研究在动态规划领域中模拟工具使用的目标，我们深入探讨了主动推断中的动态规划，该领域考虑到生物目标导向行为的两个关键方面

    

    通过动态规划，我们指的是人类大脑推断和施加与认知决策相关的运动轨迹的能力。最近的一个范式，主动推断，为生物有机体适应带来了基本见解，不断努力最小化预测误差以将自己限制在与生命兼容的状态。在过去的几年里，许多研究表明人类和动物行为可以解释为主动推断过程，无论是作为离散决策还是连续运动控制，都激发了机器人技术和人工智能中的创新解决方案。然而，文献缺乏对如何有效地在变化环境中规划行动的全面展望。我们设定了对工具使用进行建模的目标，深入研究了主动推断中的动态规划主题，牢记两个生物目标导向行为的关键方面：理解……

    arXiv:2402.11658v1 Announce Type: new  Abstract: By dynamic planning, we refer to the ability of the human brain to infer and impose motor trajectories related to cognitive decisions. A recent paradigm, active inference, brings fundamental insights into the adaptation of biological organisms, constantly striving to minimize prediction errors to restrict themselves to life-compatible states. Over the past years, many studies have shown how human and animal behavior could be explained in terms of an active inferential process -- either as discrete decision-making or continuous motor control -- inspiring innovative solutions in robotics and artificial intelligence. Still, the literature lacks a comprehensive outlook on how to effectively plan actions in changing environments. Setting ourselves the goal of modeling tool use, we delve into the topic of dynamic planning in active inference, keeping in mind two crucial aspects of biological goal-directed behavior: the capacity to understand a
    
[^3]: 在协作装配场景中，探索Cobot的生产节奏、控制定位和情绪状态之间的动态关系

    Exploring the Dynamics between Cobot's Production Rhythm, Locus of Control and Emotional State in a Collaborative Assembly Scenario

    [https://arxiv.org/abs/2402.00808](https://arxiv.org/abs/2402.00808)

    本研究探索了在协作装配场景中，Cobot的生产节奏对参与者的经验性定位控制和情绪状态的影响。虽然在情绪状态和定位控制方面未发现差异，但考虑到其他心理变量，结果表明需要考虑个体的心理特征来提供更好的互动体验。

    

    在工业场景中，协作机器人（cobots）被广泛使用，并且对评估和测量cobots的某些特征对人的影响产生了日益增长的兴趣。在本次试点研究中，研究了一个cobots的生产节奏（C1-慢、C2-快、C3-根据参与者的步调调整）对31名参与者的经验性定位控制（ELoC）和情绪状态的影响。还考虑了操作人员的绩效、基本内部定位控制程度和对机器人的态度。关于情绪状态和ELoC，在三种条件下没有发现差异，但考虑到其他心理变量，出现了更复杂的情况。总体而言，结果似乎表明需要考虑个体的心理特征，以提供不同和最佳的互动体验。

    In industrial scenarios, there is widespread use of collaborative robots (cobots), and growing interest is directed at evaluating and measuring the impact of some characteristics of the cobot on the human factor. In the present pilot study, the effect that the production rhythm (C1 - Slow, C2 - Fast, C3 - Adapted to the participant's pace) of a cobot has on the Experiential Locus of Control (ELoC) and the emotional state of 31 participants has been examined. The operators' performance, the degree of basic internal Locus of Control, and the attitude towards the robots were also considered. No difference was found regarding the emotional state and the ELoC in the three conditions, but considering the other psychological variables, a more complex situation emerges. Overall, results seem to indicate a need to consider the person's psychological characteristics to offer a differentiated and optimal interaction experience.
    

