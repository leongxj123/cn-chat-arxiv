# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DiffH2O: Diffusion-Based Synthesis of Hand-Object Interactions from Textual Descriptions](https://arxiv.org/abs/2403.17827) | 提出了一种从文本描述和物体几何形状中合成逼真的手-物体交互的方法，通过三种技术实现了有效学习，包括任务分解、紧密耦合的姿势表示和不同的引导方案。 |
| [^2] | [On Optimal Sampling for Learning SDF Using MLPs Equipped with Positional Encoding.](http://arxiv.org/abs/2401.01391) | 本文针对采样率与学习神经隐式场的准确性之间的关系进行了研究，在傅里叶分析的基础上提出了一种简单有效的方法来确定适当的采样率，以解决MLP中噪声伪影的问题。 |

# 详细

[^1]: 基于扩散的从文本描述中合成手-物体交互的方法

    DiffH2O: Diffusion-Based Synthesis of Hand-Object Interactions from Textual Descriptions

    [https://arxiv.org/abs/2403.17827](https://arxiv.org/abs/2403.17827)

    提出了一种从文本描述和物体几何形状中合成逼真的手-物体交互的方法，通过三种技术实现了有效学习，包括任务分解、紧密耦合的姿势表示和不同的引导方案。

    

    生成自然的3D手-物体交互具有挑战性，因为期望生成的手部和物体动作在物理上是合理的，并且在语义上是有意义的。我们提出了一种名为DiffH2O的新方法，可以从提供的文本提示和物体几何形状中合成逼真的单手或双手物体交互。该方法引入了三种技术，可以有效地从有限数据中学习。首先，我们将任务分解为抓取阶段和基于文本交互阶段，并为每个阶段使用单独的扩散模型。在抓取阶段中，模型仅生成手部动作，而在交互阶段中，手部和物体姿势都被合成。其次，我们提出了一种紧密耦合手部和物体姿势的紧凑表示。第三，我们提出了两种不同的引导方案。

    arXiv:2403.17827v1 Announce Type: cross  Abstract: Generating natural hand-object interactions in 3D is challenging as the resulting hand and object motions are expected to be physically plausible and semantically meaningful. Furthermore, generalization to unseen objects is hindered by the limited scale of available hand-object interaction datasets. We propose DiffH2O, a novel method to synthesize realistic, one or two-handed object interactions from provided text prompts and geometry of the object. The method introduces three techniques that enable effective learning from limited data. First, we decompose the task into a grasping stage and a text-based interaction stage and use separate diffusion models for each. In the grasping stage, the model only generates hand motions, whereas in the interaction phase both hand and object poses are synthesized. Second, we propose a compact representation that tightly couples hand and object poses. Third, we propose two different guidance schemes 
    
[^2]: 关于使用位置编码的MLP学习SDF的最优采样方法

    On Optimal Sampling for Learning SDF Using MLPs Equipped with Positional Encoding. (arXiv:2401.01391v1 [cs.CV])

    [http://arxiv.org/abs/2401.01391](http://arxiv.org/abs/2401.01391)

    本文针对采样率与学习神经隐式场的准确性之间的关系进行了研究，在傅里叶分析的基础上提出了一种简单有效的方法来确定适当的采样率，以解决MLP中噪声伪影的问题。

    

    神经隐式场，如形状的神经有符号距离场（SDF），已成为许多应用中的强大表示方法，例如编码3D形状和执行碰撞检测。通常，隐式场由带有位置编码（PE）的多层感知器（MLP）进行编码以捕捉高频几何细节。然而，这种带有PE的MLP的一个显著副作用是学习到的隐式场中存在噪声伪影。尽管增加采样率通常可以缓解这些伪影，但在本文中，我们通过傅立叶分析的视角来解释这种不良现象。我们设计了一个工具来确定学习精确神经隐式场的适当采样率，而不会产生不良的副作用。具体而言，我们提出了一种简单而有效的方法，基于网络响应的傅里叶分析，用于估计带有随机权重的给定网络的内在频率。

    Neural implicit fields, such as the neural signed distance field (SDF) of a shape, have emerged as a powerful representation for many applications, e.g., encoding a 3D shape and performing collision detection. Typically, implicit fields are encoded by Multi-layer Perceptrons (MLP) with positional encoding (PE) to capture high-frequency geometric details. However, a notable side effect of such PE-equipped MLPs is the noisy artifacts present in the learned implicit fields. While increasing the sampling rate could in general mitigate these artifacts, in this paper we aim to explain this adverse phenomenon through the lens of Fourier analysis. We devise a tool to determine the appropriate sampling rate for learning an accurate neural implicit field without undesirable side effects. Specifically, we propose a simple yet effective method to estimate the intrinsic frequency of a given network with randomized weights based on the Fourier analysis of the network's responses. It is observed that
    

