# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scene-Graph ViT: End-to-End Open-Vocabulary Visual Relationship Detection](https://arxiv.org/abs/2403.14270) | 提出了一种简单高效的无解码器架构，用于开放词汇的视觉关系检测，通过Transformer-based图像编码器隐式建模对象之间的关系，使用注意力机制提取关系信息，在混合数据上进行端到端训练，实现了最先进的关系检测性能。 |
| [^2] | [Introduction to Online Nonstochastic Control.](http://arxiv.org/abs/2211.09619) | 介绍了一种新兴的在线非随机控制方法，通过在一组策略中寻找低后悔，获得对最优策略的近似。 |

# 详细

[^1]: 场景图ViT：端到端的开放词汇视觉关系检测

    Scene-Graph ViT: End-to-End Open-Vocabulary Visual Relationship Detection

    [https://arxiv.org/abs/2403.14270](https://arxiv.org/abs/2403.14270)

    提出了一种简单高效的无解码器架构，用于开放词汇的视觉关系检测，通过Transformer-based图像编码器隐式建模对象之间的关系，使用注意力机制提取关系信息，在混合数据上进行端到端训练，实现了最先进的关系检测性能。

    

    视觉关系检测旨在识别图像中的对象及其关系。以往的方法通过在现有目标检测架构中添加单独的关系模块或解码器来处理此任务。这种分离增加了复杂性，阻碍了端到端训练，限制了性能。我们提出了一种简单且高效的无解码器架构，用于开放词汇的视觉关系检测。我们的模型由基于Transformer的图像编码器组成，将对象表示为标记，并隐含地建模它们的关系。为了提取关系信息，我们引入了一个注意力机制，选择可能形成关系的对象对。我们提供了一个单阶段的训练方法，可以在混合对象和关系检测数据上训练此模型。我们的方法在Visual Genome和大词汇GQA基准测试上实现了最先进的关系检测性能，可实现实时性。

    arXiv:2403.14270v1 Announce Type: cross  Abstract: Visual relationship detection aims to identify objects and their relationships in images. Prior methods approach this task by adding separate relationship modules or decoders to existing object detection architectures. This separation increases complexity and hinders end-to-end training, which limits performance. We propose a simple and highly efficient decoder-free architecture for open-vocabulary visual relationship detection. Our model consists of a Transformer-based image encoder that represents objects as tokens and models their relationships implicitly. To extract relationship information, we introduce an attention mechanism that selects object pairs likely to form a relationship. We provide a single-stage recipe to train this model on a mixture of object and relationship detection data. Our approach achieves state-of-the-art relationship detection performance on Visual Genome and on the large-vocabulary GQA benchmark at real-tim
    
[^2]: 在线非随机控制简介

    Introduction to Online Nonstochastic Control. (arXiv:2211.09619v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.09619](http://arxiv.org/abs/2211.09619)

    介绍了一种新兴的在线非随机控制方法，通过在一组策略中寻找低后悔，获得对最优策略的近似。

    

    本文介绍了一种新兴的动态系统控制与可微强化学习范式——在线非随机控制，并应用在线凸优化和凸松弛技术得到了具有可证明保证的新方法，在最佳和鲁棒控制方面取得了显著成果。与其他框架不同，该方法的目标是对抗性攻击，在无法预测扰动模型的情况下，通过在一组策略中寻找低后悔，获得对最优策略的近似。

    This text presents an introduction to an emerging paradigm in control of dynamical systems and differentiable reinforcement learning called online nonstochastic control. The new approach applies techniques from online convex optimization and convex relaxations to obtain new methods with provable guarantees for classical settings in optimal and robust control.  The primary distinction between online nonstochastic control and other frameworks is the objective. In optimal control, robust control, and other control methodologies that assume stochastic noise, the goal is to perform comparably to an offline optimal strategy. In online nonstochastic control, both the cost functions as well as the perturbations from the assumed dynamical model are chosen by an adversary. Thus the optimal policy is not defined a priori. Rather, the target is to attain low regret against the best policy in hindsight from a benchmark class of policies.  This objective suggests the use of the decision making frame
    

