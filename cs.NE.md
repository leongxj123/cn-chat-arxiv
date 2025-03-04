# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sine Activated Low-Rank Matrices for Parameter Efficient Learning](https://arxiv.org/abs/2403.19243) | 整合正弦函数到低秩分解过程中，提高模型准确性的同时保持参数高效性。 |
| [^2] | [Representation Learning in a Decomposed Encoder Design for Bio-inspired Hebbian Learning.](http://arxiv.org/abs/2401.08603) | 这项研究探索了在生物启发式Hebbian学习中的表示学习，并提出了一个模块化框架，利用不同的不变视觉描述符作为归纳偏见。该框架在图像分类任务上展示了较好的鲁棒性和透明度。 |

# 详细

[^1]: 用正弦激活的低秩矩阵实现参数高效学习

    Sine Activated Low-Rank Matrices for Parameter Efficient Learning

    [https://arxiv.org/abs/2403.19243](https://arxiv.org/abs/2403.19243)

    整合正弦函数到低秩分解过程中，提高模型准确性的同时保持参数高效性。

    

    低秩分解已经成为在神经网络架构中增强参数效率的重要工具，在机器学习的各种应用中越来越受到关注。这些技术显著降低了参数数量，取得了简洁性和性能之间的平衡。然而，一个常见的挑战是在参数效率和模型准确性之间做出妥协，参数减少往往导致准确性不及完整秩对应模型。在这项工作中，我们提出了一个创新的理论框架，在低秩分解过程中整合了一个正弦函数。这种方法不仅保留了低秩方法的参数效率特性的好处，还增加了分解的秩，从而提高了模型的准确性。我们的方法被证明是现有低秩模型的一种适应性增强，正如其成功证实的那样。

    arXiv:2403.19243v1 Announce Type: new  Abstract: Low-rank decomposition has emerged as a vital tool for enhancing parameter efficiency in neural network architectures, gaining traction across diverse applications in machine learning. These techniques significantly lower the number of parameters, striking a balance between compactness and performance. However, a common challenge has been the compromise between parameter efficiency and the accuracy of the model, where reduced parameters often lead to diminished accuracy compared to their full-rank counterparts. In this work, we propose a novel theoretical framework that integrates a sinusoidal function within the low-rank decomposition process. This approach not only preserves the benefits of the parameter efficiency characteristic of low-rank methods but also increases the decomposition's rank, thereby enhancing model accuracy. Our method proves to be an adaptable enhancement for existing low-rank models, as evidenced by its successful 
    
[^2]: 基于分解编码器设计的生物启发式Hebbian学习中的表示学习

    Representation Learning in a Decomposed Encoder Design for Bio-inspired Hebbian Learning. (arXiv:2401.08603v1 [cs.NE])

    [http://arxiv.org/abs/2401.08603](http://arxiv.org/abs/2401.08603)

    这项研究探索了在生物启发式Hebbian学习中的表示学习，并提出了一个模块化框架，利用不同的不变视觉描述符作为归纳偏见。该框架在图像分类任务上展示了较好的鲁棒性和透明度。

    

    现代数据驱动的机器学习系统设计利用了对架构结构的归纳偏见、不变性和等变性要求、任务特定的损失函数以及计算优化工具。先前的工作表明，编码器的早期层中的归纳偏见，以人为指定的准不变滤波器的形式，可以作为一种强大的归纳偏见，实现更好的鲁棒性和透明度。本文在生物启发式Hebbian学习的表示学习上进一步探索了这一点。我们提出了一个模块化框架，使用生物启发式的对比预测编码（Hinge CLAPP Loss）进行训练。我们的框架由多个并行编码器组成，每个编码器利用不同的不变视觉描述符作为归纳偏见。我们在不同难度的图像数据上的分类场景中评估了我们系统的表示学习能力（GTSRB, STL10, CODEBR）

    Modern data-driven machine learning system designs exploit inductive biases on architectural structure, invariance and equivariance requirements, task specific loss functions, and computational optimization tools. Previous works have illustrated that inductive bias in the early layers of the encoder in the form of human specified quasi-invariant filters can serve as a powerful inductive bias to attain better robustness and transparency in learned classifiers. This paper explores this further in the context of representation learning with local plasticity rules i.e. bio-inspired Hebbian learning . We propose a modular framework trained with a bio-inspired variant of contrastive predictive coding (Hinge CLAPP Loss). Our framework is composed of parallel encoders each leveraging a different invariant visual descriptor as an inductive bias. We evaluate the representation learning capacity of our system in a classification scenario on image data of various difficulties (GTSRB, STL10, CODEBR
    

