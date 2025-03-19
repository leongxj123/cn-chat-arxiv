# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Q-FOX Learning: Breaking Tradition in Reinforcement Learning](https://arxiv.org/abs/2402.16562) | Q-FOX学习是一种新颖的自动超参数调整方法，结合了FOX优化器和Q-learning算法，提出了使用新的目标函数来解决强化学习中超参数调整的问题。 |
| [^2] | [StochGradAdam: Accelerating Neural Networks Training with Stochastic Gradient Sampling.](http://arxiv.org/abs/2310.17042) | StochGradAdam是一种利用随机梯度抽样加速神经网络训练的优化器，通过选择性梯度考虑，能够稳定收敛，提升鲁棒训练。在图像分类和分割任务中表现优异。 |

# 详细

[^1]: Q-FOX学习：颠覆传统的强化学习

    Q-FOX Learning: Breaking Tradition in Reinforcement Learning

    [https://arxiv.org/abs/2402.16562](https://arxiv.org/abs/2402.16562)

    Q-FOX学习是一种新颖的自动超参数调整方法，结合了FOX优化器和Q-learning算法，提出了使用新的目标函数来解决强化学习中超参数调整的问题。

    

    强化学习（RL）是人工智能（AI）的一个子集，代理通过与环境的交互来学习最佳动作，因此适用于不需要标记数据或直接监督的任务。 本文提出了一种名为Q-FOX的新颖自动调参方法，该方法使用了FOX优化器和常用的易于实现的RL Q-learning算法解决了调参的问题。此外，还提出了一个新的目标函数，该函数将奖励放在均方误差（MSE）和学习时间之上。

    arXiv:2402.16562v2 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) is a subset of artificial intelligence (AI) where agents learn the best action by interacting with the environment, making it suitable for tasks that do not require labeled data or direct supervision. Hyperparameters (HP) tuning refers to choosing the best parameter that leads to optimal solutions in RL algorithms. Manual or random tuning of the HP may be a crucial process because variations in this parameter lead to changes in the overall learning aspects and different rewards. In this paper, a novel and automatic HP-tuning method called Q-FOX is proposed. This uses both the FOX optimizer, a new optimization method inspired by nature that mimics red foxes' hunting behavior, and the commonly used, easy-to-implement RL Q-learning algorithm to solve the problem of HP tuning. Moreover, a new objective function is proposed which prioritizes the reward over the mean squared error (MSE) and learning time (
    
[^2]: StochGradAdam: 利用随机梯度抽样加速神经网络训练

    StochGradAdam: Accelerating Neural Networks Training with Stochastic Gradient Sampling. (arXiv:2310.17042v1 [cs.LG])

    [http://arxiv.org/abs/2310.17042](http://arxiv.org/abs/2310.17042)

    StochGradAdam是一种利用随机梯度抽样加速神经网络训练的优化器，通过选择性梯度考虑，能够稳定收敛，提升鲁棒训练。在图像分类和分割任务中表现优异。

    

    在深度学习优化领域中，本文介绍了StochGradAdam优化器，这是对广受赞誉的Adam算法的新颖改进。StochGradAdam的核心是其梯度抽样技术。该方法不仅确保稳定收敛，而且利用选择性梯度考虑的优势，通过减轻噪声或异常数据的影响和增强损失函数空间的探索，提升了鲁棒训练。在图像分类和分割任务中，StochGradAdam表现出优于传统Adam优化器的性能。通过在每次迭代中精心选择一部分梯度进行抽样，该优化器能够有效应对复杂模型的管理。本文从数学基础到偏差校正策略全面探讨了StochGradAdam的方法，展示了深度学习训练技术的可期进展。

    In the rapidly advancing domain of deep learning optimization, this paper unveils the StochGradAdam optimizer, a novel adaptation of the well-regarded Adam algorithm. Central to StochGradAdam is its gradient sampling technique. This method not only ensures stable convergence but also leverages the advantages of selective gradient consideration, fostering robust training by potentially mitigating the effects of noisy or outlier data and enhancing the exploration of the loss landscape for more dependable convergence. In both image classification and segmentation tasks, StochGradAdam has demonstrated superior performance compared to the traditional Adam optimizer. By judiciously sampling a subset of gradients at each iteration, the optimizer is optimized for managing intricate models. The paper provides a comprehensive exploration of StochGradAdam's methodology, from its mathematical foundations to bias correction strategies, heralding a promising advancement in deep learning training tec
    

