# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DeNetDM: Debiasing by Network Depth Modulation](https://arxiv.org/abs/2403.19863) | DeNetDM 是一种基于网络深度调制的新型去偏见方法，通过使用来自专家乘积的训练范式，在创建深浅架构的偏见和去偏见分支后，将知识提炼产生目标去偏见模型，相比当前去偏见技术取得更优异的效果。 |
| [^2] | [GD doesn't make the cut: Three ways that non-differentiability affects neural network training](https://arxiv.org/abs/2401.08426) | 本文研究了非可微性对神经网络训练的影响，包括收敛性差异、$L_1$正则化问题的矛盾性质以及稳定边界现象的不适用性。 |
| [^3] | [LifelongMemory: Leveraging LLMs for Answering Queries in Long-form Egocentric Videos](https://arxiv.org/abs/2312.05269) | 终身记忆是一个新的框架，通过自然语言问答和检索方式访问长篇自我中心视频，利用零-shot能力进行推理，使用置信度和解释模块产生自信、高质量和可解释的答案，在EgoSchema问题回答基准上达到最先进性能，在Ego4D的自然语言查询挑战中具有很强的竞争力 |
| [^4] | [EMA-Net: Efficient Multitask Affinity Learning for Dense Scene Predictions.](http://arxiv.org/abs/2401.11124) | EMA-Net 是一个高效的多任务关联学习网络，通过引入跨任务关联学习模块(CTAL)，能够同时捕捉局部、全局和跨任务的相互作用。 |
| [^5] | [D3: Data Diversity Design for Systematic Generalization in Visual Question Answering.](http://arxiv.org/abs/2309.08798) | 本论文研究了视觉问答中系统一般化的关键因素，发现简单任务的多样性在实现系统一般化中起到了重要的作用，这意味着不必收集大量和多样的复杂任务。 |

# 详细

[^1]: DeNetDM: 通过网络深度调制来消除偏见

    DeNetDM: Debiasing by Network Depth Modulation

    [https://arxiv.org/abs/2403.19863](https://arxiv.org/abs/2403.19863)

    DeNetDM 是一种基于网络深度调制的新型去偏见方法，通过使用来自专家乘积的训练范式，在创建深浅架构的偏见和去偏见分支后，将知识提炼产生目标去偏见模型，相比当前去偏见技术取得更优异的效果。

    

    当神经网络在偏见数据集上训练时，它们往往会无意间学习到虚假的相关性，从而导致在实现强大的泛化性和鲁棒性方面面临挑战。目前解决这种偏见的方法通常包括利用偏见注释、根据伪偏见标签进行加权重、或通过增强技术增加偏见冲突数据点的多样性。我们引入了DeNetDM，这是一种基于观察结果的新型去偏见方法，浅层神经网络优先学习核心属性，而更深层次的神经网络在获取不同信息时强调偏见。我们利用从专家乘积中推导出的训练范式，创建了深浅架构的偏见和去偏见分支，然后用知识提炼产生目标的去偏见模型。大量实验证明，我们的方法优于当前的去偏见技术，实现了一个...

    arXiv:2403.19863v1 Announce Type: new  Abstract: When neural networks are trained on biased datasets, they tend to inadvertently learn spurious correlations, leading to challenges in achieving strong generalization and robustness. Current approaches to address such biases typically involve utilizing bias annotations, reweighting based on pseudo-bias labels, or enhancing diversity within bias-conflicting data points through augmentation techniques. We introduce DeNetDM, a novel debiasing method based on the observation that shallow neural networks prioritize learning core attributes, while deeper ones emphasize biases when tasked with acquiring distinct information. Using a training paradigm derived from Product of Experts, we create both biased and debiased branches with deep and shallow architectures and then distill knowledge to produce the target debiased model. Extensive experiments and analyses demonstrate that our approach outperforms current debiasing techniques, achieving a not
    
[^2]: GD无法胜任：非可微性对神经网络训练的三种影响方式

    GD doesn't make the cut: Three ways that non-differentiability affects neural network training

    [https://arxiv.org/abs/2401.08426](https://arxiv.org/abs/2401.08426)

    本文研究了非可微性对神经网络训练的影响，包括收敛性差异、$L_1$正则化问题的矛盾性质以及稳定边界现象的不适用性。

    

    本文研究了应用于非可微函数（NGDMs）和应用于可微函数的传统梯度下降（GDs）之间的区别。首先，我们证明了NGDMs的收敛性质与GDs存在显著差异，挑战了基于$L$-光滑性的广泛神经网络收敛文献对非光滑神经网络的适用性。接下来，我们展示了NGDM解决$L_1$正则化问题的矛盾性质，表明增加正则化惩罚会导致NGDMs中最优解的$L_1$范数增加。因此，我们证明了广泛采用的基于$L_1$惩罚的网络修剪技术并未产生预期结果。最后，我们探索了稳定边界现象（Edge of Stability），指出即使对于Lipschitz连续凸可微函数，它也不适用于非凸非可微的神经网络。

    This paper investigates the distinctions between gradient methods applied to non-differentiable functions (NGDMs) and classical gradient descents (GDs) designed for differentiable functions. First, we demonstrate significant differences in the convergence properties of NGDMs compared to GDs, challenging the applicability of the extensive neural network convergence literature based on $L-smoothness$ to non-smooth neural networks. Next, we demonstrate the paradoxical nature of NGDM solutions for $L_{1}$-regularized problems, showing that increasing the regularization penalty leads to an increase in the $L_{1}$ norm of optimal solutions in NGDMs. Consequently, we show that widely adopted $L_{1}$ penalization-based techniques for network pruning do not yield expected results. Finally, we explore the Edge of Stability phenomenon, indicating its inapplicability even to Lipschitz continuous convex differentiable functions, leaving its relevance to non-convex non-differentiable neural networks
    
[^3]: 终身记忆：利用LLMs回答长篇自我中心视频中的查询

    LifelongMemory: Leveraging LLMs for Answering Queries in Long-form Egocentric Videos

    [https://arxiv.org/abs/2312.05269](https://arxiv.org/abs/2312.05269)

    终身记忆是一个新的框架，通过自然语言问答和检索方式访问长篇自我中心视频，利用零-shot能力进行推理，使用置信度和解释模块产生自信、高质量和可解释的答案，在EgoSchema问题回答基准上达到最先进性能，在Ego4D的自然语言查询挑战中具有很强的竞争力

    

    在本文中，我们介绍了终身记忆(LifelongMemory)，这是一个新的框架，通过自然语言问答和检索来访问长篇自我中心视频存储。终身记忆生成摄像机佩戴者的简洁视频活动描述，并利用预训练的大型语言模型的零-shot能力来推理长篇视频内容。此外，终身记忆使用置信度和解释模块来产生自信、高质量和可解释的答案。我们的方法在EgoSchema问题回答基准上实现了最先进的性能，并在Ego4D的自然语言查询(NLQ)挑战中具有很强的竞争力。代码可在https://github.com/Agentic-Learning-AI-Lab/lifelong-memory 中找到。

    arXiv:2312.05269v2 Announce Type: replace-cross  Abstract: In this paper we introduce LifelongMemory, a new framework for accessing long-form egocentric videographic memory through natural language question answering and retrieval. LifelongMemory generates concise video activity descriptions of the camera wearer and leverages the zero-shot capabilities of pretrained large language models to perform reasoning over long-form video context. Furthermore, Lifelong Memory uses a confidence and explanation module to produce confident, high-quality, and interpretable answers. Our approach achieves state-of-the-art performance on the EgoSchema benchmark for question answering and is highly competitive on the natural language query (NLQ) challenge of Ego4D. Code is available at https://github.com/Agentic-Learning-AI-Lab/lifelong-memory.
    
[^4]: EMA-Net: 高效的多任务关联学习用于稠密场景预测

    EMA-Net: Efficient Multitask Affinity Learning for Dense Scene Predictions. (arXiv:2401.11124v1 [cs.CV])

    [http://arxiv.org/abs/2401.11124](http://arxiv.org/abs/2401.11124)

    EMA-Net 是一个高效的多任务关联学习网络，通过引入跨任务关联学习模块(CTAL)，能够同时捕捉局部、全局和跨任务的相互作用。

    

    多任务学习（MTL）因其能够联合预测多个任务，在使用比单任务学习更少的模型参数的情况下实现更好的每个任务性能而备受关注。最近，以解码器为重点的架构通过使用其他相关任务的特征来改进多任务性能。然而，大多数这些改进方法在以参数高效的方式同时捕捉局部和全局任务特定表示以及跨任务模式方面存在问题。在本文中，我们引入了高效多任务关联学习网络（EMA-Net），它是一个轻量级框架，增强了多任务网络的任务改进能力。EMA-Net通过我们的新颖的跨任务关联学习（CTAL）模块巧妙地捕捉局部、全局和跨任务的相互作用。CTAL的关键创新在于其能够以最适合任务亲和矩阵的方式操纵任务亲和矩阵。

    Multitask learning (MTL) has gained prominence for its ability to jointly predict multiple tasks, achieving better per-task performance while using fewer per-task model parameters than single-task learning. More recently, decoder-focused architectures have considerably improved multitask performance by refining task predictions using the features of other related tasks. However, most of these refinement methods fail to simultaneously capture local and global task-specific representations, as well as cross-task patterns in a parameter-efficient manner. In this paper, we introduce the Efficient Multitask Affinity Learning Network (EMA-Net), which is a lightweight framework that enhances the task refinement capabilities of multitask networks. EMA-Net adeptly captures local, global, and cross-task interactions using our novel Cross-Task Affinity Learning (CTAL) module. The key innovation of CTAL lies in its ability to manipulate task affinity matrices in a manner that is optimally suited t
    
[^5]: D3: 数据多样性设计为系统一般化在视觉问答中。

    D3: Data Diversity Design for Systematic Generalization in Visual Question Answering. (arXiv:2309.08798v1 [cs.AI])

    [http://arxiv.org/abs/2309.08798](http://arxiv.org/abs/2309.08798)

    本论文研究了视觉问答中系统一般化的关键因素，发现简单任务的多样性在实现系统一般化中起到了重要的作用，这意味着不必收集大量和多样的复杂任务。

    

    系统一般化是智能的关键方面，它指的是通过结合已知的子任务和概念来推广到新任务的能力。已经显示影响系统一般化的一个关键因素是训练数据的多样性。然而，多样性可以以多种方式定义，因为数据具有许多变化因素。对于不同方面的数据多样性如何影响系统一般化的更细致的理解尚缺乏。我们在视觉问答（VQA）问题中提供了新的证据，揭示了简单任务的多样性（即由几个子任务和概念组成的任务）在实现系统一般化中的关键作用。这意味着收集大量和多样化的复杂任务可能并非必要，这可能成本高昂。我们证明了这个结果与训练和测试数据之间的相似性无关，并适用于众所周知的神经网络家族。

    Systematic generalization is a crucial aspect of intelligence, which refers to the ability to generalize to novel tasks by combining known subtasks and concepts. One critical factor that has been shown to influence systematic generalization is the diversity of training data. However, diversity can be defined in various ways, as data have many factors of variation. A more granular understanding of how different aspects of data diversity affect systematic generalization is lacking. We present new evidence in the problem of Visual Question Answering (VQA) that reveals that the diversity of simple tasks (i.e. tasks formed by a few subtasks and concepts) plays a key role in achieving systematic generalization. This implies that it may not be essential to gather a large and varied number of complex tasks, which could be costly to obtain. We demonstrate that this result is independent of the similarity between the training and testing data and applies to well-known families of neural network 
    

