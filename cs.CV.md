# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SLEDGE: Synthesizing Simulation Environments for Driving Agents with Generative Models](https://arxiv.org/abs/2403.17933) | SLEDGE是第一个使用生成模型训练的车辆运动规划生成模拟器，引入了新颖的栅格到矢量自编码器（RVAE）以及Diffusion Transformer来生成智能体和车道图，从而实现更好的模拟控制。 |
| [^2] | [Deep Learning Safety Concerns in Automated Driving Perception.](http://arxiv.org/abs/2309.03774) | 本研究旨在通过引入安全考虑作为结构元素，以系统综合的方式确保基于深度神经网络的自动驾驶系统的安全性。这一概念不仅与现有的安全标准相契合，还为AI安全相关的学术出版物和标准提供了新的启示。 |
| [^3] | [Generalized Continual Category Discovery.](http://arxiv.org/abs/2308.12112) | 本研究提出了一种广义持续类别发现（GCCD）的框架，用于在现实生活场景中同时处理新的和已知的类别，并且利用持续的无监督学习方法来发现它们。通过实验证明现有方法无法处理后续任务中的无标记样本。 |
| [^4] | [Leveraging Large Language Models for Scalable Vector Graphics-Driven Image Understanding.](http://arxiv.org/abs/2306.06094) | 本文介绍了一种新方法，通过使用可伸缩矢量图(SVG)格式处理图像，使得大型语言模型(LLMs)能够直接理解和操作图像，而无需使用参数化的视觉组件。在图像分类、生成和上下文学习等任务上，该方法表现良好，具有鲁棒性和显著的性能提升。 |
| [^5] | [Fairness-aware Vision Transformer via Debiased Self-Attention.](http://arxiv.org/abs/2301.13803) | 这篇论文提出了一种基于去偏自注意力的公平感知视觉变换器框架，通过消除与敏感属性相关的虚假特征来减轻偏见，并利用对抗性示例来定位和屏蔽这些特征。 |

# 详细

[^1]: SLEDGE: 使用生成模型合成驾驶智能体的模拟环境

    SLEDGE: Synthesizing Simulation Environments for Driving Agents with Generative Models

    [https://arxiv.org/abs/2403.17933](https://arxiv.org/abs/2403.17933)

    SLEDGE是第一个使用生成模型训练的车辆运动规划生成模拟器，引入了新颖的栅格到矢量自编码器（RVAE）以及Diffusion Transformer来生成智能体和车道图，从而实现更好的模拟控制。

    

    SLEDGE是第一个基于真实世界驾驶记录训练的车辆运动规划生成模拟器。其核心组件是一个学习模型，能够生成智能体边界框和车道图。该模型的输出作为交通模拟的初始状态。针对SLEDGE待生成的实体的独特特性，例如它们的连接性和每个场景的可变数量，使得大多数现代生成模型在这一任务上的朴素应用变得不简单。因此，我们除了对现有车道图表示进行系统研究外，还引入了一种新颖的栅格到矢量自编码器（RVAE）。它将智能体和车道图编码为栅格化潜在映射中的不同通道。这有助于车道条件下的智能体生成以及使用扩散变换器同时生成车道和智能体。在SLEDGE中使用生成的实体可以更好地控制模拟，例如上采样转弯。

    arXiv:2403.17933v1 Announce Type: cross  Abstract: SLEDGE is the first generative simulator for vehicle motion planning trained on real-world driving logs. Its core component is a learned model that is able to generate agent bounding boxes and lane graphs. The model's outputs serve as an initial state for traffic simulation. The unique properties of the entities to be generated for SLEDGE, such as their connectivity and variable count per scene, render the naive application of most modern generative models to this task non-trivial. Therefore, together with a systematic study of existing lane graph representations, we introduce a novel raster-to-vector autoencoder (RVAE). It encodes agents and the lane graph into distinct channels in a rasterized latent map. This facilitates both lane-conditioned agent generation and combined generation of lanes and agents with a Diffusion Transformer. Using generated entities in SLEDGE enables greater control over the simulation, e.g. upsampling turns 
    
[^2]: 自动驾驶感知中的深度学习安全考虑

    Deep Learning Safety Concerns in Automated Driving Perception. (arXiv:2309.03774v1 [cs.LG])

    [http://arxiv.org/abs/2309.03774](http://arxiv.org/abs/2309.03774)

    本研究旨在通过引入安全考虑作为结构元素，以系统综合的方式确保基于深度神经网络的自动驾驶系统的安全性。这一概念不仅与现有的安全标准相契合，还为AI安全相关的学术出版物和标准提供了新的启示。

    

    深度学习领域的最新进展以及深度神经网络（DNNs）在感知方面的出色性能导致了对其在自动驾驶系统中应用的增加需求。这类系统的安全性至关重要，因此需要考虑DNNs的独特属性。为了以系统综合的方式确保基于DNNs的自动驾驶系统的安全性，引入了所谓的安全考虑作为适当的结构元素。一方面，安全考虑的概念设计与现有的与自动驾驶系统安全相关的标准如ISO 21448（SOTIF）非常契合。另一方面，它已经激发了几篇学术出版物和即将出台的关于AI安全的标准，如ISO PAS 8800。虽然安全考虑的概念以前已经被介绍过，但本文对其进行了扩展和优化，借鉴了各个领域和安全专家的反馈意见。

    Recent advances in the field of deep learning and impressive performance of deep neural networks (DNNs) for perception have resulted in an increased demand for their use in automated driving (AD) systems. The safety of such systems is of utmost importance and thus requires to consider the unique properties of DNNs.  In order to achieve safety of AD systems with DNN-based perception components in a systematic and comprehensive approach, so-called safety concerns have been introduced as a suitable structuring element. On the one hand, the concept of safety concerns is -- by design -- well aligned to existing standards relevant for safety of AD systems such as ISO 21448 (SOTIF). On the other hand, it has already inspired several academic publications and upcoming standards on AI safety such as ISO PAS 8800.  While the concept of safety concerns has been previously introduced, this paper extends and refines it, leveraging feedback from various domain and safety experts in the field. In par
    
[^3]: 广义持续类别发现

    Generalized Continual Category Discovery. (arXiv:2308.12112v1 [cs.LG])

    [http://arxiv.org/abs/2308.12112](http://arxiv.org/abs/2308.12112)

    本研究提出了一种广义持续类别发现（GCCD）的框架，用于在现实生活场景中同时处理新的和已知的类别，并且利用持续的无监督学习方法来发现它们。通过实验证明现有方法无法处理后续任务中的无标记样本。

    

    大多数持续学习（CL）方法推动着监督学习设置的极限，其中一个智能体期望学习新的标记任务而不会忘记先前的知识。然而，这些设置与现实生活场景不太吻合，其中学习智能体可以访问大量的无标记数据，包括全新（完全无标记）类别和已知类别的示例。受到广义类别发现（GCD）的启发，我们引入了一个新的框架来放松这个假设。确切地说，在任何任务中，我们允许存在新的和已知的类别，并且必须使用持续版本的无监督学习方法来发现它们。我们称这种设置为广义持续类别发现（GCCD）。它统一了CL和GCD，弥合了合成基准和现实生活场景之间的差距。通过一系列实验，我们发现现有的方法无法从后续任务中积累知识，其中包含无标记样本。

    Most of Continual Learning (CL) methods push the limit of supervised learning settings, where an agent is expected to learn new labeled tasks and not forget previous knowledge. However, these settings are not well aligned with real-life scenarios, where a learning agent has access to a vast amount of unlabeled data encompassing both novel (entirely unlabeled) classes and examples from known classes. Drawing inspiration from Generalized Category Discovery (GCD), we introduce a novel framework that relaxes this assumption. Precisely, in any task, we allow for the existence of novel and known classes, and one must use continual version of unsupervised learning methods to discover them. We call this setting Generalized Continual Category Discovery (GCCD). It unifies CL and GCD, bridging the gap between synthetic benchmarks and real-life scenarios. With a series of experiments, we present that existing methods fail to accumulate knowledge from subsequent tasks in which unlabeled samples of 
    
[^4]: 利用大型语言模型进行可伸缩矢量图驱动的图像理解

    Leveraging Large Language Models for Scalable Vector Graphics-Driven Image Understanding. (arXiv:2306.06094v1 [cs.CV])

    [http://arxiv.org/abs/2306.06094](http://arxiv.org/abs/2306.06094)

    本文介绍了一种新方法，通过使用可伸缩矢量图(SVG)格式处理图像，使得大型语言模型(LLMs)能够直接理解和操作图像，而无需使用参数化的视觉组件。在图像分类、生成和上下文学习等任务上，该方法表现良好，具有鲁棒性和显著的性能提升。

    

    最近，大型语言模型(LLMs)在自然语言理解和生成方面取得了显著进展。然而，它们在计算机视觉领域的潜力仍然很大程度上没有被开发。本文介绍了一种新的探索性方法，使得LLMs能够使用可伸缩矢量图(SVG)格式处理图像。通过利用基于XML的SVG表述的文本描述而不是光栅图像，我们旨在弥合视觉和文本模态之间的差距，使LLMs能够直接理解和操作图像，而无需使用参数化的视觉组件。我们的方法仅利用LLMs的能力进行简单的图像分类、生成和上下文学习。我们展示了方法在判别和生成任务中的优异表现，重点介绍了它(i)对分布转移的鲁棒性，(ii)通过利用LLMs的上下文学习能力实现的显著改进，以及(iii)图像杂乱程度上的显著性能提高。

    Recently, large language models (LLMs) have made significant advancements in natural language understanding and generation. However, their potential in computer vision remains largely unexplored. In this paper, we introduce a new, exploratory approach that enables LLMs to process images using the Scalable Vector Graphics (SVG) format. By leveraging the XML-based textual descriptions of SVG representations instead of raster images, we aim to bridge the gap between the visual and textual modalities, allowing LLMs to directly understand and manipulate images without the need for parameterized visual components. Our method facilitates simple image classification, generation, and in-context learning using only LLM capabilities. We demonstrate the promise of our approach across discriminative and generative tasks, highlighting its (i) robustness against distribution shift, (ii) substantial improvements achieved by tapping into the in-context learning abilities of LLMs, and (iii) image unders
    
[^5]: 基于去偏自注意力的公平感知视觉变换器

    Fairness-aware Vision Transformer via Debiased Self-Attention. (arXiv:2301.13803v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2301.13803](http://arxiv.org/abs/2301.13803)

    这篇论文提出了一种基于去偏自注意力的公平感知视觉变换器框架，通过消除与敏感属性相关的虚假特征来减轻偏见，并利用对抗性示例来定位和屏蔽这些特征。

    

    最近，由于其提取信息特征和通过自我关注机制建模长距离依赖关系的能力，视觉变换器（ViT）在解决计算机视觉（CV）问题方面引起了广泛关注。为了充分发挥ViT在现实应用中的优势，最近的研究探索了ViT的可靠性和可解释性，包括其鲁棒性和可解释性。然而，另一个需求，公平性，在文献中尚未得到充分解决。我们证明了现有的公平感知算法（主要设计用于CNN）在ViT上表现不佳。这就需要我们通过去偏自注意（DSA）开发我们的新框架。DSA是一种通过盲目方法来强制ViT消除与敏感属性相关的虚假特征以减轻偏见的方法。值得注意的是，对抗性示例被用来定位和屏蔽输入图像块中的虚假特征。

    Vision Transformer (ViT) has recently gained significant interest in solving computer vision (CV) problems due to its capability of extracting informative features and modeling long-range dependencies through the self-attention mechanism. To fully realize the advantages of ViT in real-world applications, recent works have explored the trustworthiness of ViT, including its robustness and explainability. However, another desiderata, fairness has not yet been adequately addressed in the literature. We establish that the existing fairness-aware algorithms (primarily designed for CNNs) do not perform well on ViT. This necessitates the need for developing our novel framework via Debiased Self-Attention (DSA). DSA is a fairness-through-blindness approach that enforces ViT to eliminate spurious features correlated with the sensitive attributes for bias mitigation. Notably, adversarial examples are leveraged to locate and mask the spurious features in the input image patches. In addition, DSA u
    

