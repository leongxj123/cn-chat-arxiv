# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Traffic Signal Control via Genetic Programming](https://arxiv.org/abs/2403.17328) | 提出了一种新的基于学习的方法用于解决复杂交叉路口的信号控制问题，通过设计阶段紧急性概念和可解释的树结构，可以在信号转换期间选择激活的信号相位。 |
| [^2] | [Data augmentation with automated machine learning: approaches and performance comparison with classical data augmentation methods](https://arxiv.org/abs/2403.08352) | 自动化机器学习的数据增强方法旨在自动化数据增强过程，为改善机器学习模型泛化性能提供了更高效的方式。 |

# 详细

[^1]: 通过遗传编程学习交通信号控制

    Learning Traffic Signal Control via Genetic Programming

    [https://arxiv.org/abs/2403.17328](https://arxiv.org/abs/2403.17328)

    提出了一种新的基于学习的方法用于解决复杂交叉路口的信号控制问题，通过设计阶段紧急性概念和可解释的树结构，可以在信号转换期间选择激活的信号相位。

    

    交通信号控制对提高交通效率至关重要。最近，基于学习的方法，特别是深度强化学习（DRL），在寻求更有效的交通信号控制策略方面取得了巨大成功。然而，在DRL中奖励的设计高度依赖领域知识才能收敛到有效策略，而最终策略也存在解释困难。在本工作中，提出了一种新的面向复杂路口的信号控制的学习方法。在我们的方法中，我们为每个信号相设计了一个阶段紧急性的概念。在信号变换期间，交通灯控制策略根据阶段紧急性选择要激活的下一个相位。然后，我们提出将紧急功能表示为可解释的树结构。紧急功能可以根据当前道路条件为特定相位计算相位紧急性。

    arXiv:2403.17328v1 Announce Type: new  Abstract: The control of traffic signals is crucial for improving transportation efficiency. Recently, learning-based methods, especially Deep Reinforcement Learning (DRL), garnered substantial success in the quest for more efficient traffic signal control strategies. However, the design of rewards in DRL highly demands domain knowledge to converge to an effective policy, and the final policy also presents difficulties in terms of explainability. In this work, a new learning-based method for signal control in complex intersections is proposed. In our approach, we design a concept of phase urgency for each signal phase. During signal transitions, the traffic light control strategy selects the next phase to be activated based on the phase urgency. We then proposed to represent the urgency function as an explainable tree structure. The urgency function can calculate the phase urgency for a specific phase based on the current road conditions. Genetic 
    
[^2]: 利用自动化机器学习的数据增强方法及与传统数据增强方法性能比较

    Data augmentation with automated machine learning: approaches and performance comparison with classical data augmentation methods

    [https://arxiv.org/abs/2403.08352](https://arxiv.org/abs/2403.08352)

    自动化机器学习的数据增强方法旨在自动化数据增强过程，为改善机器学习模型泛化性能提供了更高效的方式。

    

    数据增强被认为是常用于提高机器学习模型泛化性能的最重要的正则化技术。它主要涉及应用适当的数据转换操作，以创建具有所需属性的新数据样本。尽管其有效性，这一过程通常具有挑战性，因为手动创建和测试不同候选增强及其超参数需耗费大量时间。自动化数据增强方法旨在自动化这一过程。最先进的方法通常依赖于自动化机器学习（AutoML）原则。本研究提供了基于AutoML的数据增强技术的全面调查。我们讨论了使用AutoML实现数据增强的各种方法，包括数据操作、数据集成和数据合成技术。我们详细讨论了技术

    arXiv:2403.08352v1 Announce Type: cross  Abstract: Data augmentation is arguably the most important regularization technique commonly used to improve generalization performance of machine learning models. It primarily involves the application of appropriate data transformation operations to create new data samples with desired properties. Despite its effectiveness, the process is often challenging because of the time-consuming trial and error procedures for creating and testing different candidate augmentations and their hyperparameters manually. Automated data augmentation methods aim to automate the process. State-of-the-art approaches typically rely on automated machine learning (AutoML) principles. This work presents a comprehensive survey of AutoML-based data augmentation techniques. We discuss various approaches for accomplishing data augmentation with AutoML, including data manipulation, data integration and data synthesis techniques. We present extensive discussion of technique
    

