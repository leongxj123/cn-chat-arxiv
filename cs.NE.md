# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Library of Mirrors: Deep Neural Nets in Low Dimensions are Convex Lasso Models with Reflection Features](https://arxiv.org/abs/2403.01046) | 证明在1-D数据上训练神经网络等价于解决一个具有固定特征字典矩阵的凸Lasso问题，为全局最优网络和解空间提供了洞察。 |
| [^2] | [A match made in consistency heaven: when large language models meet evolutionary algorithms.](http://arxiv.org/abs/2401.10510) | 大型语言模型和进化算法的结合具有强大的一致性，包括标记嵌入和基因型-表现型映射、位置编码和适应性塑造、位置嵌入和选择、注意力和交叉、前馈神经网络和突变、模型训练和参数更新以及多任务学习和多目标优化等多个核心特征。本文分析了现有的耦合研究，并为未来的研究提供了基本路线和关键挑战。 |
| [^3] | [SpikeCP: Delay-Adaptive Reliable Spiking Neural Networks via Conformal Prediction.](http://arxiv.org/abs/2305.11322) | 这篇论文提出了一种新的脉冲神经网络模型，能够通过极限预测实现自适应的推断延迟，从而节约能源与提高可靠性。 |
| [^4] | [Learning Symbolic Model-Agnostic Loss Functions via Meta-Learning.](http://arxiv.org/abs/2209.08907) | 本文提出了一种通过元学习框架学习模型无关损失函数的方法，并通过对多个监督学习任务的实验证明，该方法学到的损失函数优于目前最优方法和交叉熵损失函数。 |

# 详细

[^1]: 一个镜子的库：低维深度神经网络是具有反射特征的凸Lasso模型

    A Library of Mirrors: Deep Neural Nets in Low Dimensions are Convex Lasso Models with Reflection Features

    [https://arxiv.org/abs/2403.01046](https://arxiv.org/abs/2403.01046)

    证明在1-D数据上训练神经网络等价于解决一个具有固定特征字典矩阵的凸Lasso问题，为全局最优网络和解空间提供了洞察。

    

    我们证明在1-D数据上训练神经网络等价于解决一个带有固定、明确定义的特征字典矩阵的凸Lasso问题。具体的字典取决于激活函数和深度。我们考虑具有分段线性激活函数的两层网络，深窄的ReLU网络最多有4层，以及具有符号激活和任意深度的矩形和树网络。有趣的是，在ReLU网络中，第四层创建代表训练数据关于自身的反射的特征。Lasso表示法揭示了全局最优网络和解空间的洞察。

    arXiv:2403.01046v1 Announce Type: cross  Abstract: We prove that training neural networks on 1-D data is equivalent to solving a convex Lasso problem with a fixed, explicitly defined dictionary matrix of features. The specific dictionary depends on the activation and depth. We consider 2-layer networks with piecewise linear activations, deep narrow ReLU networks with up to 4 layers, and rectangular and tree networks with sign activation and arbitrary depth. Interestingly in ReLU networks, a fourth layer creates features that represent reflections of training data about themselves. The Lasso representation sheds insight to globally optimal networks and the solution landscape.
    
[^2]: 天作之合：大型语言模型与进化算法的结合

    A match made in consistency heaven: when large language models meet evolutionary algorithms. (arXiv:2401.10510v1 [cs.NE])

    [http://arxiv.org/abs/2401.10510](http://arxiv.org/abs/2401.10510)

    大型语言模型和进化算法的结合具有强大的一致性，包括标记嵌入和基因型-表现型映射、位置编码和适应性塑造、位置嵌入和选择、注意力和交叉、前馈神经网络和突变、模型训练和参数更新以及多任务学习和多目标优化等多个核心特征。本文分析了现有的耦合研究，并为未来的研究提供了基本路线和关键挑战。

    

    预训练的大型语言模型（LLMs）在生成创造性的自然文本方面具有强大的能力。进化算法（EAs）可以发现复杂实际问题的多样解决方案。本文通过比较文本序列生成和进化的共同特点和方向性，阐述了LLMs与EAs之间的强大一致性，包括多个一对一的核心特征：标记嵌入和基因型-表现型映射、位置编码和适应性塑造、位置嵌入和选择、注意力和交叉、前馈神经网络和突变、模型训练和参数更新以及多任务学习和多目标优化。在这种一致性视角下，分析了现有的耦合研究，包括进化微调和LLM增强型EAs。借助这些洞见，我们概述了未来在LLMs和EAs耦合方面的基本研究路线，并突出了其中的关键挑战。

    Pre-trained large language models (LLMs) have powerful capabilities for generating creative natural text. Evolutionary algorithms (EAs) can discover diverse solutions to complex real-world problems. Motivated by the common collective and directionality of text sequence generation and evolution, this paper illustrates the strong consistency of LLMs and EAs, which includes multiple one-to-one key characteristics: token embedding and genotype-phenotype mapping, position encoding and fitness shaping, position embedding and selection, attention and crossover, feed-forward neural network and mutation, model training and parameter update, and multi-task learning and multi-objective optimization. Based on this consistency perspective, existing coupling studies are analyzed, including evolutionary fine-tuning and LLM-enhanced EAs. Leveraging these insights, we outline a fundamental roadmap for future research in coupling LLMs and EAs, while highlighting key challenges along the way. The consist
    
[^3]: SpikeCP: 通过极限预测实现延迟自适应可靠脉冲神经网络

    SpikeCP: Delay-Adaptive Reliable Spiking Neural Networks via Conformal Prediction. (arXiv:2305.11322v1 [cs.NE])

    [http://arxiv.org/abs/2305.11322](http://arxiv.org/abs/2305.11322)

    这篇论文提出了一种新的脉冲神经网络模型，能够通过极限预测实现自适应的推断延迟，从而节约能源与提高可靠性。

    

    脉冲神经网络（SNN）通过内部事件驱动的神经动态处理时间序列数据，其能量消耗取决于输入演示期间神经元之间交换的脉冲数量。在典型的SNN分类器实现中，决策是在整个输入序列被处理后产生的，导致延迟和能量消耗水平在输入之间是相对均匀的。最近引入的延迟自适应SNN可根据每个示例的难度来定制推断延迟 - 以及随之而来的能耗 - 通过在SNN模型足够“自信”时产生早期决策来实现。

    Spiking neural networks (SNNs) process time-series data via internal event-driven neural dynamics whose energy consumption depends on the number of spikes exchanged between neurons over the course of the input presentation. In typical implementations of an SNN classifier, decisions are produced after the entire input sequence has been processed, resulting in latency and energy consumption levels that are fairly uniform across inputs. Recently introduced delay-adaptive SNNs tailor the inference latency -- and, with it, the energy consumption -- to the difficulty of each example, by producing an early decision when the SNN model is sufficiently ``confident''. In this paper, we start by observing that, as an SNN processes input samples, its classification decisions tend to be first under-confident and then over-confident with respect to the decision's ground-truth, unknown, test accuracy. This makes it difficult to determine a stopping time that ensures a desired level of accuracy. To add
    
[^4]: 通过元学习学习符号模型无关损失函数

    Learning Symbolic Model-Agnostic Loss Functions via Meta-Learning. (arXiv:2209.08907v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.08907](http://arxiv.org/abs/2209.08907)

    本文提出了一种通过元学习框架学习模型无关损失函数的方法，并通过对多个监督学习任务的实验证明，该方法学到的损失函数优于目前最优方法和交叉熵损失函数。

    

    本文研究损失函数学习的新兴主题，旨在学习可以显著提高模型性能的损失函数。我们提出了一种新的元学习框架，通过混合神经符号搜索方法学习模型无关的损失函数。该框架首先使用基于进化的方法在原始数学操作空间中搜索符号损失函数的集合。然后，学习到的一组损失函数通过端到端的梯度训练过程进行参数化和优化。所提出的框架的多功能性在一组多样化的监督学习任务上得到了经验证实。结果显示，新提出的方法发现的元学习损失函数在各种神经网络架构和数据集上均优于交叉熵损失和现有最先进的损失函数学习方法。

    In this paper, we develop upon the emerging topic of loss function learning, which aims to learn loss functions that significantly improve the performance of the models trained under them. Specifically, we propose a new meta-learning framework for learning model-agnostic loss functions via a hybrid neuro-symbolic search approach. The framework first uses evolution-based methods to search the space of primitive mathematical operations to find a set of symbolic loss functions. Second, the set of learned loss functions are subsequently parameterized and optimized via an end-to-end gradient-based training procedure. The versatility of the proposed framework is empirically validated on a diverse set of supervised learning tasks. Results show that the meta-learned loss functions discovered by the newly proposed method outperform both the cross-entropy loss and state-of-the-art loss function learning methods on a diverse range of neural network architectures and datasets.
    

