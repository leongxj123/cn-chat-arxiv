# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Artificial consciousness. Some logical and conceptual preliminaries](https://arxiv.org/abs/2403.20177) | 需要在人工系统中平衡讨论意识的可能实现，提出了使用意识的维度和特征来进行讨论的必要性。 |
| [^2] | [Speeding Up Path Planning via Reinforcement Learning in MCTS for Automated Parking](https://arxiv.org/abs/2403.17234) | 本文提出了一种将强化学习与蒙特卡洛树搜索集成的方法，用于自动停车任务中的在线路径规划，旨在加速路径规划过程，提高效率。 |
| [^3] | [EC-IoU: Orienting Safety for Object Detectors via Ego-Centric Intersection-over-Union](https://arxiv.org/abs/2403.15474) | 通过EC-IoU度量，本文引入了一种定向安全性物体检测方法，可以在安全关键领域中提高物体检测器的性能，并在KITTI数据集上取得了比IoU更好的结果。 |

# 详细

[^1]: 人工意识。一些逻辑和概念初步

    Artificial consciousness. Some logical and conceptual preliminaries

    [https://arxiv.org/abs/2403.20177](https://arxiv.org/abs/2403.20177)

    需要在人工系统中平衡讨论意识的可能实现，提出了使用意识的维度和特征来进行讨论的必要性。

    

    arXiv:2403.20177v1 公告类型: 新的 摘要: 人工意识在理论上是否可能？是否合乎情理？如果是，那么技术上可行吗？要解决这些问题，有必要奠定一些基础，阐明人工意识产生的逻辑和经验条件以及涉及的相关术语的含义。意识是一个多义词：来自不同领域的研究人员，包括神经科学、人工智能、机器人技术和哲学等，有时会使用不同术语来指称相同现象，或者使用相同术语来指称不同现象。事实上，如果我们想探讨人工意识，就需要恰当界定关键概念。在此，经过一些逻辑和概念初步工作后，我们认为有必要使用意识的维度和特征进行平衡讨论，探讨它们在人工系统中的可能实例化或实现。我们在这项工作的主要目标是...

    arXiv:2403.20177v1 Announce Type: new  Abstract: Is artificial consciousness theoretically possible? Is it plausible? If so, is it technically feasible? To make progress on these questions, it is necessary to lay some groundwork clarifying the logical and empirical conditions for artificial consciousness to arise and the meaning of relevant terms involved. Consciousness is a polysemic word: researchers from different fields, including neuroscience, Artificial Intelligence, robotics, and philosophy, among others, sometimes use different terms in order to refer to the same phenomena or the same terms to refer to different phenomena. In fact, if we want to pursue artificial consciousness, a proper definition of the key concepts is required. Here, after some logical and conceptual preliminaries, we argue for the necessity of using dimensions and profiles of consciousness for a balanced discussion about their possible instantiation or realisation in artificial systems. Our primary goal in t
    
[^2]: 在自动停车中通过强化学习在MCTS中加速路径规划

    Speeding Up Path Planning via Reinforcement Learning in MCTS for Automated Parking

    [https://arxiv.org/abs/2403.17234](https://arxiv.org/abs/2403.17234)

    本文提出了一种将强化学习与蒙特卡洛树搜索集成的方法，用于自动停车任务中的在线路径规划，旨在加速路径规划过程，提高效率。

    

    本文针对一种方法进行了讨论，该方法将强化学习整合到蒙特卡洛树搜索中，以提升在全可观测环境下进行自动停车任务的在线路径规划。在高维空间下基于采样的规划方法可能具有计算开销大、耗时长的特点。状态评估方法通过将先验知识应用于搜索步骤中，使实时系统中的过程更快速。鉴于自动停车任务通常在复杂环境中执行，传统分析方式难以构建坚实但轻量级的启发式指导。为了克服这一局限性，我们提出了在路径规划框架下具有蒙特卡洛树搜索的强化学习流水线。通过迭代地学习状态的价值以及最佳动作，在前一个周期结果的样本中选择最佳动作，我们能够建模一个值估计器以及一个...

    arXiv:2403.17234v1 Announce Type: new  Abstract: In this paper, we address a method that integrates reinforcement learning into the Monte Carlo tree search to boost online path planning under fully observable environments for automated parking tasks. Sampling-based planning methods under high-dimensional space can be computationally expensive and time-consuming. State evaluation methods are useful by leveraging the prior knowledge into the search steps, making the process faster in a real-time system. Given the fact that automated parking tasks are often executed under complex environments, a solid but lightweight heuristic guidance is challenging to compose in a traditional analytical way. To overcome this limitation, we propose a reinforcement learning pipeline with a Monte Carlo tree search under the path planning framework. By iteratively learning the value of a state and the best action among samples from its previous cycle's outcomes, we are able to model a value estimator and a 
    
[^3]: EC-IoU: 通过自我中心交并联调整物体检测器的安全性

    EC-IoU: Orienting Safety for Object Detectors via Ego-Centric Intersection-over-Union

    [https://arxiv.org/abs/2403.15474](https://arxiv.org/abs/2403.15474)

    通过EC-IoU度量，本文引入了一种定向安全性物体检测方法，可以在安全关键领域中提高物体检测器的性能，并在KITTI数据集上取得了比IoU更好的结果。

    

    本文介绍了通过一种新颖的自我中心交并联（EC-IoU）度量来定向安全性物体检测，解决了在自动驾驶等安全关键领域应用最先进的基于学习的感知模型时面临的实际问题。具体来说，我们提出了一种加权机制来优化广泛使用的IoU度量，使其能够根据自我代理人的视角覆盖更近的地面真实对象点的预测分配更高的分数。所提出的EC-IoU度量可以用于典型的评估过程，选择有更高安全性表现的物体检测器用于下游任务。它还可以集成到常见损失函数中进行模型微调。尽管面向安全性，但我们在KITTI数据集上的实验表明，使用EC-IoU训练的模型在均值平均精度方面的性能可能会优于使用IoU训练的变体。

    arXiv:2403.15474v1 Announce Type: cross  Abstract: This paper presents safety-oriented object detection via a novel Ego-Centric Intersection-over-Union (EC-IoU) measure, addressing practical concerns when applying state-of-the-art learning-based perception models in safety-critical domains such as autonomous driving. Concretely, we propose a weighting mechanism to refine the widely used IoU measure, allowing it to assign a higher score to a prediction that covers closer points of a ground-truth object from the ego agent's perspective. The proposed EC-IoU measure can be used in typical evaluation processes to select object detectors with higher safety-related performance for downstream tasks. It can also be integrated into common loss functions for model fine-tuning. While geared towards safety, our experiment with the KITTI dataset demonstrates the performance of a model trained on EC-IoU can be better than that of a variant trained on IoU in terms of mean Average Precision as well.
    

