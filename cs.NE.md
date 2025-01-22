# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Recurrent Transformers with Dynamic Halt](https://rss.arxiv.org/abs/2402.00976) | 本文研究了增强Transformer与循环机制的两种方法，并提出了新的扩展和组合方法。在多个诊断任务中进行比较，探索它们的归纳偏好。 |
| [^2] | [Q-FOX Learning: Breaking Tradition in Reinforcement Learning](https://arxiv.org/abs/2402.16562) | Q-FOX学习是一种新颖的自动超参数调整方法，结合了FOX优化器和Q-learning算法，提出了使用新的目标函数来解决强化学习中超参数调整的问题。 |

# 详细

[^1]: 具有动态停止的循环Transformer

    Recurrent Transformers with Dynamic Halt

    [https://rss.arxiv.org/abs/2402.00976](https://rss.arxiv.org/abs/2402.00976)

    本文研究了增强Transformer与循环机制的两种方法，并提出了新的扩展和组合方法。在多个诊断任务中进行比较，探索它们的归纳偏好。

    

    本文研究了两种主要方法在增强Transformer与循环机制方面的归纳偏好——（1）类似于Universal Transformers的深度逐层循环方法；和（2）类似于Temporal Latent Bottleneck的分块时态循环方法。此外，我们提出并研究了扩展和组合上述方法的新方式，例如，我们提出了一种基于全局均值的Universal Transformer动态停止机制，并将Universal Transformer的元素融入到Temporal Latent Bottleneck中。我们通过多个诊断任务（如Long Range Arena（LRA），翻转-翻转语言建模，ListOps和逻辑推理）比较了模型并探索了它们的归纳偏好。

    In this paper, we study the inductive biases of two major approaches to augmenting Transformers with a recurrent mechanism - (1) the approach of incorporating a depth-wise recurrence similar to Universal Transformers; and (2) the approach of incorporating a chunk-wise temporal recurrence like Temporal Latent Bottleneck. Furthermore, we propose and investigate novel ways to extend and combine the above methods - for example, we propose a global mean-based dynamic halting mechanism for Universal Transformer and an augmentation of Temporal Latent Bottleneck with elements from Universal Transformer. We compare the models and probe their inductive biases in several diagnostic tasks such as Long Range Arena (LRA), flip-flop language modeling, ListOps, and Logical Inference.
    
[^2]: Q-FOX学习：颠覆传统的强化学习

    Q-FOX Learning: Breaking Tradition in Reinforcement Learning

    [https://arxiv.org/abs/2402.16562](https://arxiv.org/abs/2402.16562)

    Q-FOX学习是一种新颖的自动超参数调整方法，结合了FOX优化器和Q-learning算法，提出了使用新的目标函数来解决强化学习中超参数调整的问题。

    

    强化学习（RL）是人工智能（AI）的一个子集，代理通过与环境的交互来学习最佳动作，因此适用于不需要标记数据或直接监督的任务。 本文提出了一种名为Q-FOX的新颖自动调参方法，该方法使用了FOX优化器和常用的易于实现的RL Q-learning算法解决了调参的问题。此外，还提出了一个新的目标函数，该函数将奖励放在均方误差（MSE）和学习时间之上。

    arXiv:2402.16562v2 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) is a subset of artificial intelligence (AI) where agents learn the best action by interacting with the environment, making it suitable for tasks that do not require labeled data or direct supervision. Hyperparameters (HP) tuning refers to choosing the best parameter that leads to optimal solutions in RL algorithms. Manual or random tuning of the HP may be a crucial process because variations in this parameter lead to changes in the overall learning aspects and different rewards. In this paper, a novel and automatic HP-tuning method called Q-FOX is proposed. This uses both the FOX optimizer, a new optimization method inspired by nature that mimics red foxes' hunting behavior, and the commonly used, easy-to-implement RL Q-learning algorithm to solve the problem of HP tuning. Moreover, a new objective function is proposed which prioritizes the reward over the mean squared error (MSE) and learning time (
    

