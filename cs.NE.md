# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sine Activated Low-Rank Matrices for Parameter Efficient Learning](https://arxiv.org/abs/2403.19243) | 整合正弦函数到低秩分解过程中，提高模型准确性的同时保持参数高效性。 |
| [^2] | [Brain-inspired Distributed Memorization Learning for Efficient Feature-free Unsupervised Domain Adaptation](https://arxiv.org/abs/2402.14598) | 提出了一种受到人类大脑记忆机制启发的分布式记忆学习机制，通过随机连接的神经元记忆输入信号的关联，并基于置信度关联分布式记忆，能够在无需特征微调的情况下，通过强化记忆适应新领域，适合部署在边缘设备上。 |
| [^3] | [Life-inspired Interoceptive Artificial Intelligence for Autonomous and Adaptive Agents.](http://arxiv.org/abs/2309.05999) | 该论文提出了一种基于生命理论和控制论的新视角，通过将控制论、强化学习和神经科学的最新进展与生命理论相结合，将内感知应用于构建具有自主和适应性能力的人工智能代理。 |

# 详细

[^1]: 用正弦激活的低秩矩阵实现参数高效学习

    Sine Activated Low-Rank Matrices for Parameter Efficient Learning

    [https://arxiv.org/abs/2403.19243](https://arxiv.org/abs/2403.19243)

    整合正弦函数到低秩分解过程中，提高模型准确性的同时保持参数高效性。

    

    低秩分解已经成为在神经网络架构中增强参数效率的重要工具，在机器学习的各种应用中越来越受到关注。这些技术显著降低了参数数量，取得了简洁性和性能之间的平衡。然而，一个常见的挑战是在参数效率和模型准确性之间做出妥协，参数减少往往导致准确性不及完整秩对应模型。在这项工作中，我们提出了一个创新的理论框架，在低秩分解过程中整合了一个正弦函数。这种方法不仅保留了低秩方法的参数效率特性的好处，还增加了分解的秩，从而提高了模型的准确性。我们的方法被证明是现有低秩模型的一种适应性增强，正如其成功证实的那样。

    arXiv:2403.19243v1 Announce Type: new  Abstract: Low-rank decomposition has emerged as a vital tool for enhancing parameter efficiency in neural network architectures, gaining traction across diverse applications in machine learning. These techniques significantly lower the number of parameters, striking a balance between compactness and performance. However, a common challenge has been the compromise between parameter efficiency and the accuracy of the model, where reduced parameters often lead to diminished accuracy compared to their full-rank counterparts. In this work, we propose a novel theoretical framework that integrates a sinusoidal function within the low-rank decomposition process. This approach not only preserves the benefits of the parameter efficiency characteristic of low-rank methods but also increases the decomposition's rank, thereby enhancing model accuracy. Our method proves to be an adaptable enhancement for existing low-rank models, as evidenced by its successful 
    
[^2]: 基于大脑启发的分布式记忆学习用于高效的无特征自动适应领域

    Brain-inspired Distributed Memorization Learning for Efficient Feature-free Unsupervised Domain Adaptation

    [https://arxiv.org/abs/2402.14598](https://arxiv.org/abs/2402.14598)

    提出了一种受到人类大脑记忆机制启发的分布式记忆学习机制，通过随机连接的神经元记忆输入信号的关联，并基于置信度关联分布式记忆，能够在无需特征微调的情况下，通过强化记忆适应新领域，适合部署在边缘设备上。

    

    与基于梯度的人工神经网络相比，生物神经网络通常表现出更强大的泛化能力，能够快速适应未知环境而无需使用任何梯度反向传播程序。受人类大脑分布式记忆机制的启发，我们提出了一种新颖的基于梯度的分布式记忆学习机制，称为DML，以支持转移模型的快速领域适应。具体来说，DML采用随机连接的神经元来记忆输入信号的关联，这些信号作为冲动传播，并通过关联分布式记忆的置信度做出最终决策。更重要的是，DML能够基于未标记数据进行强化记忆，快速适应新领域，而无需对深层特征进行繁重的微调，这使其非常适合部署在边缘设备上。基于四个交叉领域的真实世界实验。

    arXiv:2402.14598v1 Announce Type: cross  Abstract: Compared with gradient based artificial neural networks, biological neural networks usually show a more powerful generalization ability to quickly adapt to unknown environments without using any gradient back-propagation procedure. Inspired by the distributed memory mechanism of human brains, we propose a novel gradient-free Distributed Memorization Learning mechanism, namely DML, to support quick domain adaptation of transferred models. In particular, DML adopts randomly connected neurons to memorize the association of input signals, which are propagated as impulses, and makes the final decision by associating the distributed memories based on their confidence. More importantly, DML is able to perform reinforced memorization based on unlabeled data to quickly adapt to a new domain without heavy fine-tuning of deep features, which makes it very suitable for deploying on edge devices. Experiments based on four cross-domain real-world da
    
[^3]: 生命启发的自主和适应智能为自主和适应性代理构建具有自主能力和自适应能力的代理一直是人工智能（AI）的终极目标。生物体是这样一个代理的最好例证，它为自适应自主性提供了重要的经验教训。在这里，我们关注内感知，这是一个监控自身内部环境来保持在一定范围内的过程，它为生物体的生存提供了基础。为了开发具有内感知的AI，我们需要将表示内部环境的状态变量与外部环境相分离，并采用生命启发的内部环境状态的数学特性。本文提出了一个新的视角，即通过将控制论、强化学习和神经科学的最新进展与生命理论相结合，内感知如何帮助构建自主和适应性代理。

    Life-inspired Interoceptive Artificial Intelligence for Autonomous and Adaptive Agents. (arXiv:2309.05999v1 [cs.AI])

    [http://arxiv.org/abs/2309.05999](http://arxiv.org/abs/2309.05999)

    该论文提出了一种基于生命理论和控制论的新视角，通过将控制论、强化学习和神经科学的最新进展与生命理论相结合，将内感知应用于构建具有自主和适应性能力的人工智能代理。

    

    构建具有自主能力和自适应能力的代理一直是人工智能（AI）的终极目标。生物体是这样一个代理的最好例证，它为自适应自主性提供了重要的经验教训。本文关注内感知，这是一个监控自身内部环境来保持在一定范围内的过程，它为生物体的生存提供了基础。为了开发具有内感知的AI，我们需要将表示内部环境的状态变量与外部环境相分离，并采用生命启发的内部环境状态的数学特性。本文提出了一个新的视角，即通过将控制论、强化学习和神经科学的最新进展与生命理论相结合，内感知如何帮助构建自主和适应性代理。

    Building autonomous --- i.e., choosing goals based on one's needs -- and adaptive -- i.e., surviving in ever-changing environments -- agents has been a holy grail of artificial intelligence (AI). A living organism is a prime example of such an agent, offering important lessons about adaptive autonomy. Here, we focus on interoception, a process of monitoring one's internal environment to keep it within certain bounds, which underwrites the survival of an organism. To develop AI with interoception, we need to factorize the state variables representing internal environments from external environments and adopt life-inspired mathematical properties of internal environment states. This paper offers a new perspective on how interoception can help build autonomous and adaptive agents by integrating the legacy of cybernetics with recent advances in theories of life, reinforcement learning, and neuroscience.
    

