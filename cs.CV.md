# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DiffH2O: Diffusion-Based Synthesis of Hand-Object Interactions from Textual Descriptions](https://arxiv.org/abs/2403.17827) | 提出了一种从文本描述和物体几何形状中合成逼真的手-物体交互的方法，通过三种技术实现了有效学习，包括任务分解、紧密耦合的姿势表示和不同的引导方案。 |
| [^2] | [Continual Vision-and-Language Navigation](https://arxiv.org/abs/2403.15049) | 该论文提出了持续视觉和语言导航（CVLN）范式，旨在解决现有训练VLN代理方法固有的固定数据集的重大限制，使代理能够在不断变化的真实世界中进行导航。 |
| [^3] | [Recurrent Aligned Network for Generalized Pedestrian Trajectory Prediction](https://arxiv.org/abs/2403.05810) | 引入了循环对齐网络（RAN）来最小化领域差异，通过循环对齐策略有效地在时间-状态和时间-序列级别对齐轨迹特征空间，从而实现广义行人轨迹预测。 |
| [^4] | [On Optimal Sampling for Learning SDF Using MLPs Equipped with Positional Encoding.](http://arxiv.org/abs/2401.01391) | 本文针对采样率与学习神经隐式场的准确性之间的关系进行了研究，在傅里叶分析的基础上提出了一种简单有效的方法来确定适当的采样率，以解决MLP中噪声伪影的问题。 |
| [^5] | [Upgrading VAE Training With Unlimited Data Plans Provided by Diffusion Models.](http://arxiv.org/abs/2310.19653) | 这项研究通过在预训练的扩散模型生成的样本上进行训练，有效减轻了VAE中编码器的过拟合问题。 |

# 详细

[^1]: 基于扩散的从文本描述中合成手-物体交互的方法

    DiffH2O: Diffusion-Based Synthesis of Hand-Object Interactions from Textual Descriptions

    [https://arxiv.org/abs/2403.17827](https://arxiv.org/abs/2403.17827)

    提出了一种从文本描述和物体几何形状中合成逼真的手-物体交互的方法，通过三种技术实现了有效学习，包括任务分解、紧密耦合的姿势表示和不同的引导方案。

    

    生成自然的3D手-物体交互具有挑战性，因为期望生成的手部和物体动作在物理上是合理的，并且在语义上是有意义的。我们提出了一种名为DiffH2O的新方法，可以从提供的文本提示和物体几何形状中合成逼真的单手或双手物体交互。该方法引入了三种技术，可以有效地从有限数据中学习。首先，我们将任务分解为抓取阶段和基于文本交互阶段，并为每个阶段使用单独的扩散模型。在抓取阶段中，模型仅生成手部动作，而在交互阶段中，手部和物体姿势都被合成。其次，我们提出了一种紧密耦合手部和物体姿势的紧凑表示。第三，我们提出了两种不同的引导方案。

    arXiv:2403.17827v1 Announce Type: cross  Abstract: Generating natural hand-object interactions in 3D is challenging as the resulting hand and object motions are expected to be physically plausible and semantically meaningful. Furthermore, generalization to unseen objects is hindered by the limited scale of available hand-object interaction datasets. We propose DiffH2O, a novel method to synthesize realistic, one or two-handed object interactions from provided text prompts and geometry of the object. The method introduces three techniques that enable effective learning from limited data. First, we decompose the task into a grasping stage and a text-based interaction stage and use separate diffusion models for each. In the grasping stage, the model only generates hand motions, whereas in the interaction phase both hand and object poses are synthesized. Second, we propose a compact representation that tightly couples hand and object poses. Third, we propose two different guidance schemes 
    
[^2]: Continual Vision-and-Language Navigation

    Continual Vision-and-Language Navigation

    [https://arxiv.org/abs/2403.15049](https://arxiv.org/abs/2403.15049)

    该论文提出了持续视觉和语言导航（CVLN）范式，旨在解决现有训练VLN代理方法固有的固定数据集的重大限制，使代理能够在不断变化的真实世界中进行导航。

    

    视觉和语言导航（VLN）代理根据自然语言指令和观察到的视觉信息导航到目的地。现有的VLN代理训练方法预设固定数据集，导致一个重大限制：引入新环境需要重新训练以保留已经遇到的环境的知识。这使得在不断变化的真实世界中训练VLN代理变得困难。为了解决这一限制，我们提出了持续视觉和语言导航（CVLN）范式，旨在通过一个持续学习过程评估代理。

    arXiv:2403.15049v1 Announce Type: cross  Abstract: Vision-and-Language Navigation (VLN) agents navigate to a destination using natural language instructions and the visual information they observe. Existing methods for training VLN agents presuppose fixed datasets, leading to a significant limitation: the introduction of new environments necessitates retraining with previously encountered environments to preserve their knowledge. This makes it difficult to train VLN agents that operate in the ever-changing real world. To address this limitation, we present the Continual Vision-and-Language Navigation (CVLN) paradigm, designed to evaluate agents trained through a continual learning process. For the training and evaluation of CVLN agents, we re-arrange existing VLN datasets to propose two datasets: CVLN-I, focused on navigation via initial-instruction interpretation, and CVLN-D, aimed at navigation through dialogue with other agents. Furthermore, we propose two novel rehearsal-based meth
    
[^3]: 用于广义行人轨迹预测的循环对齐网络

    Recurrent Aligned Network for Generalized Pedestrian Trajectory Prediction

    [https://arxiv.org/abs/2403.05810](https://arxiv.org/abs/2403.05810)

    引入了循环对齐网络（RAN）来最小化领域差异，通过循环对齐策略有效地在时间-状态和时间-序列级别对齐轨迹特征空间，从而实现广义行人轨迹预测。

    

    行人轨迹预测在计算机视觉和机器人领域中是一个关键组成部分，但由于领域转移问题而仍然具有挑战性。以往的研究试图通过利用来自目标领域的部分轨迹数据来调整模型来解决这个问题。然而，在现实世界的场景中，这些领域自适应方法是不切实际的，因为不太可能从所有潜在的目标领域收集轨迹数据。本文研究了一项名为广义行人轨迹预测的任务，旨在将模型推广到看不见的领域，而无需访问它们的轨迹。为了解决这个任务，我们引入了一个循环对齐网络（RAN）来通过领域对齐来最小化领域差异。具体地，我们设计了一个循环对齐模块，通过循环对齐策略有效地在时间-状态和时间-序列级别对齐轨迹特征空间。

    arXiv:2403.05810v1 Announce Type: cross  Abstract: Pedestrian trajectory prediction is a crucial component in computer vision and robotics, but remains challenging due to the domain shift problem. Previous studies have tried to tackle this problem by leveraging a portion of the trajectory data from the target domain to adapt the model. However, such domain adaptation methods are impractical in real-world scenarios, as it is infeasible to collect trajectory data from all potential target domains. In this paper, we study a task named generalized pedestrian trajectory prediction, with the aim of generalizing the model to unseen domains without accessing their trajectories. To tackle this task, we introduce a Recurrent Aligned Network~(RAN) to minimize the domain gap through domain alignment. Specifically, we devise a recurrent alignment module to effectively align the trajectory feature spaces at both time-state and time-sequence levels by the recurrent alignment strategy.Furthermore, we 
    
[^4]: 关于使用位置编码的MLP学习SDF的最优采样方法

    On Optimal Sampling for Learning SDF Using MLPs Equipped with Positional Encoding. (arXiv:2401.01391v1 [cs.CV])

    [http://arxiv.org/abs/2401.01391](http://arxiv.org/abs/2401.01391)

    本文针对采样率与学习神经隐式场的准确性之间的关系进行了研究，在傅里叶分析的基础上提出了一种简单有效的方法来确定适当的采样率，以解决MLP中噪声伪影的问题。

    

    神经隐式场，如形状的神经有符号距离场（SDF），已成为许多应用中的强大表示方法，例如编码3D形状和执行碰撞检测。通常，隐式场由带有位置编码（PE）的多层感知器（MLP）进行编码以捕捉高频几何细节。然而，这种带有PE的MLP的一个显著副作用是学习到的隐式场中存在噪声伪影。尽管增加采样率通常可以缓解这些伪影，但在本文中，我们通过傅立叶分析的视角来解释这种不良现象。我们设计了一个工具来确定学习精确神经隐式场的适当采样率，而不会产生不良的副作用。具体而言，我们提出了一种简单而有效的方法，基于网络响应的傅里叶分析，用于估计带有随机权重的给定网络的内在频率。

    Neural implicit fields, such as the neural signed distance field (SDF) of a shape, have emerged as a powerful representation for many applications, e.g., encoding a 3D shape and performing collision detection. Typically, implicit fields are encoded by Multi-layer Perceptrons (MLP) with positional encoding (PE) to capture high-frequency geometric details. However, a notable side effect of such PE-equipped MLPs is the noisy artifacts present in the learned implicit fields. While increasing the sampling rate could in general mitigate these artifacts, in this paper we aim to explain this adverse phenomenon through the lens of Fourier analysis. We devise a tool to determine the appropriate sampling rate for learning an accurate neural implicit field without undesirable side effects. Specifically, we propose a simple yet effective method to estimate the intrinsic frequency of a given network with randomized weights based on the Fourier analysis of the network's responses. It is observed that
    
[^5]: 使用扩散模型提供的无限数据计划升级VAE训练

    Upgrading VAE Training With Unlimited Data Plans Provided by Diffusion Models. (arXiv:2310.19653v1 [stat.ML])

    [http://arxiv.org/abs/2310.19653](http://arxiv.org/abs/2310.19653)

    这项研究通过在预训练的扩散模型生成的样本上进行训练，有效减轻了VAE中编码器的过拟合问题。

    

    变分自编码器（VAE）是一种常用的表示学习模型，但其编码器容易过拟合，因为它们是在有限的训练集上进行训练，而不是真实（连续）数据分布$p_{\mathrm{data}}(\mathbf{x})$。与之相反，扩散模型通过固定编码器避免了这个问题。这使得它们的表示不太可解释，但简化了训练，可以精确和连续地逼近$p_{\mathrm{data}}(\mathbf{x})$。在本文中，我们展示了通过在预训练的扩散模型生成的样本上训练，可以有效减轻VAE中编码器的过拟合问题。这些结果有些出人意料，因为最近的研究发现，在使用另一个生成模型生成的数据上训练时，生成性能会下降。我们分析了使用我们的方法训练的VAE的泛化性能、分摊差距和鲁棒性。

    Variational autoencoders (VAEs) are popular models for representation learning but their encoders are susceptible to overfitting (Cremer et al., 2018) because they are trained on a finite training set instead of the true (continuous) data distribution $p_{\mathrm{data}}(\mathbf{x})$. Diffusion models, on the other hand, avoid this issue by keeping the encoder fixed. This makes their representations less interpretable, but it simplifies training, enabling accurate and continuous approximations of $p_{\mathrm{data}}(\mathbf{x})$. In this paper, we show that overfitting encoders in VAEs can be effectively mitigated by training on samples from a pre-trained diffusion model. These results are somewhat unexpected as recent findings (Alemohammad et al., 2023; Shumailov et al., 2023) observe a decay in generative performance when models are trained on data generated by another generative model. We analyze generalization performance, amortization gap, and robustness of VAEs trained with our pro
    

