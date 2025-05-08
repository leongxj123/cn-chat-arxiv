# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Opening Cabinets and Drawers in the Real World using a Commodity Mobile Manipulator](https://arxiv.org/abs/2402.17767) | 实现了一个端到端系统，使商品移动操作器成功在以前未见的真实世界环境中打开橱柜和抽屉，感知误差是主要挑战。 |
| [^2] | [Navigating Neural Space: Revisiting Concept Activation Vectors to Overcome Directional Divergence](https://arxiv.org/abs/2202.03482) | 本文重新审视了概念激活向量（CAVs）在建模人类可理解的概念中的应用，并引入了基于模式的CAVs来提供更准确的概念方向。 |

# 详细

[^1]: 在现实世界中使用商品移动操作器打开橱柜和抽屉

    Opening Cabinets and Drawers in the Real World using a Commodity Mobile Manipulator

    [https://arxiv.org/abs/2402.17767](https://arxiv.org/abs/2402.17767)

    实现了一个端到端系统，使商品移动操作器成功在以前未见的真实世界环境中打开橱柜和抽屉，感知误差是主要挑战。

    

    在这项工作中，我们构建了一个端到端系统，使商品移动操作器（Stretch RE2）能够在多样的以前未见的真实世界环境中拉开橱柜和抽屉。我们在31个不同的物体和13个不同真实世界环境中进行了4天的实际测试。我们的系统在零击打下，对在未知环境中新颖的橱柜和抽屉的打开率达到61%。对失败模式的分析表明，感知误差是我们系统面临的最重要挑战。

    arXiv:2402.17767v1 Announce Type: cross  Abstract: Pulling open cabinets and drawers presents many difficult technical challenges in perception (inferring articulation parameters for objects from onboard sensors), planning (producing motion plans that conform to tight task constraints), and control (making and maintaining contact while applying forces on the environment). In this work, we build an end-to-end system that enables a commodity mobile manipulator (Stretch RE2) to pull open cabinets and drawers in diverse previously unseen real world environments. We conduct 4 days of real world testing of this system spanning 31 different objects from across 13 different real world environments. Our system achieves a success rate of 61% on opening novel cabinets and drawers in unseen environments zero-shot. An analysis of the failure modes suggests that errors in perception are the most significant challenge for our system. We will open source code and models for others to replicate and bui
    
[^2]: 领航神经空间：重新审视概念激活向量以克服方向差异

    Navigating Neural Space: Revisiting Concept Activation Vectors to Overcome Directional Divergence

    [https://arxiv.org/abs/2202.03482](https://arxiv.org/abs/2202.03482)

    本文重新审视了概念激活向量（CAVs）在建模人类可理解的概念中的应用，并引入了基于模式的CAVs来提供更准确的概念方向。

    

    随着对于理解神经网络预测策略的兴趣日益增长，概念激活向量（CAVs）已成为一种流行的工具，用于在潜在空间中建模人类可理解的概念。通常，CAVs是通过利用线性分类器来计算的，该分类器优化具有给定概念和无给定概念的样本的潜在表示的可分离性。然而，在本文中我们展示了这种以可分离性为导向的计算方法会导致与精确建模概念方向的实际目标发散的解决方案。这种差异可以归因于分散方向的显著影响，即与概念无关的信号，这些信号被线性模型的滤波器（即权重）捕获以优化类别可分性。为了解决这个问题，我们引入基于模式的CAVs，仅关注概念信号，从而提供更准确的概念方向。我们评估了各种CAV方法与真实概念方向的对齐程度。

    With a growing interest in understanding neural network prediction strategies, Concept Activation Vectors (CAVs) have emerged as a popular tool for modeling human-understandable concepts in the latent space. Commonly, CAVs are computed by leveraging linear classifiers optimizing the separability of latent representations of samples with and without a given concept. However, in this paper we show that such a separability-oriented computation leads to solutions, which may diverge from the actual goal of precisely modeling the concept direction. This discrepancy can be attributed to the significant influence of distractor directions, i.e., signals unrelated to the concept, which are picked up by filters (i.e., weights) of linear models to optimize class-separability. To address this, we introduce pattern-based CAVs, solely focussing on concept signals, thereby providing more accurate concept directions. We evaluate various CAV methods in terms of their alignment with the true concept dire
    

