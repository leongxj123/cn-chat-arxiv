# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Hybrid SNN-ANN Network for Event-based Object Detection with Spatial and Temporal Attention](https://arxiv.org/abs/2403.10173) | 提出了一种用于基于事件的对象检测的混合SNN-ANN网络，包括了新颖的基于注意力的桥接模块，能够有效捕捉稀疏的空间和时间关系，以提高任务性能。 |
| [^2] | [Sustainable Transparency in Recommender Systems: Bayesian Ranking of Images for Explainability.](http://arxiv.org/abs/2308.01196) | 这项研究旨在实现推荐系统的可持续透明性，并提出了一种使用贝叶斯排名图像进行个性化解释的方法，以最大化透明度和用户信任。 |

# 详细

[^1]: 一种用于基于事件的对象检测的混合SNN-ANN网络，具有空间和时间注意力机制

    A Hybrid SNN-ANN Network for Event-based Object Detection with Spatial and Temporal Attention

    [https://arxiv.org/abs/2403.10173](https://arxiv.org/abs/2403.10173)

    提出了一种用于基于事件的对象检测的混合SNN-ANN网络，包括了新颖的基于注意力的桥接模块，能够有效捕捉稀疏的空间和时间关系，以提高任务性能。

    

    事件相机提供高时间分辨率和动态范围，几乎没有运动模糊，非常适合对象检测任务。尖峰神经网络（SNN）与事件驱动感知数据天生匹配，在神经形态硬件上能够实现超低功耗和低延迟推断，而人工神经网络（ANN）则展示出更稳定的训练动态和更快的收敛速度，从而具有更好的任务性能。混合SNN-ANN方法是一种有前途的替代方案，能够利用SNN和ANN体系结构的优势。在这项工作中，我们引入了第一个基于混合注意力的SNN-ANN骨干网络，用于使用事件相机进行对象检测。我们提出了一种新颖的基于注意力的SNN-ANN桥接模块，从SNN层中捕捉稀疏的空间和时间关系，并将其转换为密集特征图，供骨干网络的ANN部分使用。实验结果表明，我们提出的m

    arXiv:2403.10173v1 Announce Type: cross  Abstract: Event cameras offer high temporal resolution and dynamic range with minimal motion blur, making them promising for object detection tasks. While Spiking Neural Networks (SNNs) are a natural match for event-based sensory data and enable ultra-energy efficient and low latency inference on neuromorphic hardware, Artificial Neural Networks (ANNs) tend to display more stable training dynamics and faster convergence resulting in greater task performance. Hybrid SNN-ANN approaches are a promising alternative, enabling to leverage the strengths of both SNN and ANN architectures. In this work, we introduce the first Hybrid Attention-based SNN-ANN backbone for object detection using event cameras. We propose a novel Attention-based SNN-ANN bridge module to capture sparse spatial and temporal relations from the SNN layer and convert them into dense feature maps for the ANN part of the backbone. Experimental results demonstrate that our proposed m
    
[^2]: 可持续透明的推荐系统: 用于解释性的贝叶斯图像排名

    Sustainable Transparency in Recommender Systems: Bayesian Ranking of Images for Explainability. (arXiv:2308.01196v1 [cs.IR])

    [http://arxiv.org/abs/2308.01196](http://arxiv.org/abs/2308.01196)

    这项研究旨在实现推荐系统的可持续透明性，并提出了一种使用贝叶斯排名图像进行个性化解释的方法，以最大化透明度和用户信任。

    

    推荐系统在现代世界中变得至关重要，通常指导用户找到相关的内容或产品，并对用户和公民的决策产生重大影响。然而，确保这些系统的透明度和用户信任仍然是一个挑战；个性化解释已经成为一个解决方案，为推荐提供理由。在生成个性化解释的现有方法中，使用用户创建的视觉内容是一个特别有潜力的选项，有潜力最大化透明度和用户信任。然而，现有模型在这个背景下解释推荐时存在一些限制：可持续性是一个关键问题，因为它们经常需要大量的计算资源，导致的碳排放量与它们被整合到推荐系统中相当。此外，大多数模型使用的替代学习目标与排名最有效的目标不一致。

    Recommender Systems have become crucial in the modern world, commonly guiding users towards relevant content or products, and having a large influence over the decisions of users and citizens. However, ensuring transparency and user trust in these systems remains a challenge; personalized explanations have emerged as a solution, offering justifications for recommendations. Among the existing approaches for generating personalized explanations, using visual content created by the users is one particularly promising option, showing a potential to maximize transparency and user trust. Existing models for explaining recommendations in this context face limitations: sustainability has been a critical concern, as they often require substantial computational resources, leading to significant carbon emissions comparable to the Recommender Systems where they would be integrated. Moreover, most models employ surrogate learning goals that do not align with the objective of ranking the most effect
    

