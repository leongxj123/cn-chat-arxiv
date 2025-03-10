# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Universal Post-Training Reverse-Engineering Defense Against Backdoors in Deep Neural Networks](https://arxiv.org/abs/2402.02034) | 本文提出了一种针对深度神经网络中后门攻击的通用后训练反向工程防御方法，通过依赖内部特征图来检测和反向工程后门，并识别其目标类别，具有广泛适用性和低计算开销。 |
| [^2] | [A match made in consistency heaven: when large language models meet evolutionary algorithms.](http://arxiv.org/abs/2401.10510) | 大型语言模型和进化算法的结合具有强大的一致性，包括标记嵌入和基因型-表现型映射、位置编码和适应性塑造、位置嵌入和选择、注意力和交叉、前馈神经网络和突变、模型训练和参数更新以及多任务学习和多目标优化等多个核心特征。本文分析了现有的耦合研究，并为未来的研究提供了基本路线和关键挑战。 |
| [^3] | [A fuzzy adaptive evolutionary-based feature selection and machine learning framework for single and multi-objective body fat prediction.](http://arxiv.org/abs/2303.11949) | 本文提出了一种模糊自适应进化特征选择与机器学习框架用于单目标和多目标身体脂肪预测，该方法在管理参数化和计算成本的同时确定了适当的探索和开发水平，可以避免陷入局部最优问题。 |

# 详细

[^1]: 深度神经网络中针对后门攻击的通用后训练反向工程防御方法

    Universal Post-Training Reverse-Engineering Defense Against Backdoors in Deep Neural Networks

    [https://arxiv.org/abs/2402.02034](https://arxiv.org/abs/2402.02034)

    本文提出了一种针对深度神经网络中后门攻击的通用后训练反向工程防御方法，通过依赖内部特征图来检测和反向工程后门，并识别其目标类别，具有广泛适用性和低计算开销。

    

    针对深度神经网络分类器的后门攻击，提出了各种防御方法。通用方法旨在可靠地检测和/或减轻后门攻击，而反向工程方法通常明确假设其中一种。本文提出了一种新的检测器，它依赖于被防守的DNN的内部特征图来检测和反向工程后门，并识别其目标类别；它可以在后训练时操作（无需访问训练数据集）；对于不同的嵌入机制（即通用的）非常有效；并且具有低计算开销，因此可扩展。我们对基准CIFAR-10图像分类器的不同攻击进行了检测方法的评估。

    A variety of defenses have been proposed against backdoors attacks on deep neural network (DNN) classifiers. Universal methods seek to reliably detect and/or mitigate backdoors irrespective of the incorporation mechanism used by the attacker, while reverse-engineering methods often explicitly assume one. In this paper, we describe a new detector that: relies on internal feature map of the defended DNN to detect and reverse-engineer the backdoor and identify its target class; can operate post-training (without access to the training dataset); is highly effective for various incorporation mechanisms (i.e., is universal); and which has low computational overhead and so is scalable. Our detection approach is evaluated for different attacks on a benchmark CIFAR-10 image classifier.
    
[^2]: 天作之合：大型语言模型与进化算法的结合

    A match made in consistency heaven: when large language models meet evolutionary algorithms. (arXiv:2401.10510v1 [cs.NE])

    [http://arxiv.org/abs/2401.10510](http://arxiv.org/abs/2401.10510)

    大型语言模型和进化算法的结合具有强大的一致性，包括标记嵌入和基因型-表现型映射、位置编码和适应性塑造、位置嵌入和选择、注意力和交叉、前馈神经网络和突变、模型训练和参数更新以及多任务学习和多目标优化等多个核心特征。本文分析了现有的耦合研究，并为未来的研究提供了基本路线和关键挑战。

    

    预训练的大型语言模型（LLMs）在生成创造性的自然文本方面具有强大的能力。进化算法（EAs）可以发现复杂实际问题的多样解决方案。本文通过比较文本序列生成和进化的共同特点和方向性，阐述了LLMs与EAs之间的强大一致性，包括多个一对一的核心特征：标记嵌入和基因型-表现型映射、位置编码和适应性塑造、位置嵌入和选择、注意力和交叉、前馈神经网络和突变、模型训练和参数更新以及多任务学习和多目标优化。在这种一致性视角下，分析了现有的耦合研究，包括进化微调和LLM增强型EAs。借助这些洞见，我们概述了未来在LLMs和EAs耦合方面的基本研究路线，并突出了其中的关键挑战。

    Pre-trained large language models (LLMs) have powerful capabilities for generating creative natural text. Evolutionary algorithms (EAs) can discover diverse solutions to complex real-world problems. Motivated by the common collective and directionality of text sequence generation and evolution, this paper illustrates the strong consistency of LLMs and EAs, which includes multiple one-to-one key characteristics: token embedding and genotype-phenotype mapping, position encoding and fitness shaping, position embedding and selection, attention and crossover, feed-forward neural network and mutation, model training and parameter update, and multi-task learning and multi-objective optimization. Based on this consistency perspective, existing coupling studies are analyzed, including evolutionary fine-tuning and LLM-enhanced EAs. Leveraging these insights, we outline a fundamental roadmap for future research in coupling LLMs and EAs, while highlighting key challenges along the way. The consist
    
[^3]: 一种模糊自适应进化特征选择与机器学习框架用于单目标和多目标身体脂肪预测

    A fuzzy adaptive evolutionary-based feature selection and machine learning framework for single and multi-objective body fat prediction. (arXiv:2303.11949v1 [cs.NE])

    [http://arxiv.org/abs/2303.11949](http://arxiv.org/abs/2303.11949)

    本文提出了一种模糊自适应进化特征选择与机器学习框架用于单目标和多目标身体脂肪预测，该方法在管理参数化和计算成本的同时确定了适当的探索和开发水平，可以避免陷入局部最优问题。

    

    预测身体脂肪可以为医学从业者和用户提供预防和诊断心脏疾病的重要信息。混合机器学习模型通过选择相关的身体测量值和捕捉模型中所选特征之间的复杂非线性关系，提供了比简单的回归分析方法更好的性能。然而，它们仍然存在一些缺点。当前的机器学习建模方法将身体脂肪预测问题建模为组合的单目标和多目标优化问题，往往会陷入局部最优。当多个特征子集产生类似或接近的预测时，避免局部最优变得更加复杂。进化特征选择已被用于解决几个基于机器学习的优化问题。一个模糊集理论确定适当的探索和开发水平，同时管理参数化和计算成本。采用加权和身体脂肪预测方法进行实验评估

    Predicting body fat can provide medical practitioners and users with essential information for preventing and diagnosing heart diseases. Hybrid machine learning models offer better performance than simple regression analysis methods by selecting relevant body measurements and capturing complex nonlinear relationships among selected features in modelling body fat prediction problems. There are, however, some disadvantages to them. Current machine learning. Modelling body fat prediction as a combinatorial single- and multi-objective optimisation problem often gets stuck in local optima. When multiple feature subsets produce similar or close predictions, avoiding local optima becomes more complex. Evolutionary feature selection has been used to solve several machine-learning-based optimisation problems. A fuzzy set theory determines appropriate levels of exploration and exploitation while managing parameterisation and computational costs. A weighted-sum body fat prediction approach was ex
    

