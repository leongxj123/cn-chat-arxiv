# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PICL: Physics Informed Contrastive Learning for Partial Differential Equations](https://arxiv.org/abs/2401.16327) | 这项工作开发了一种使用广义对比损失的对比预训练框架，通过利用物理信息改善了神经算子在多个偏微分方程中的泛化能力。 |
| [^2] | [Adaptive joint distribution learning.](http://arxiv.org/abs/2110.04829) | 该论文提出了一种自适应联合分布学习的框架，可以从大量数据点中估计低维、归一化和正的Radon-Nikodym导数模型，并在不同学习问题上取得了良好的结果。 |

# 详细

[^1]: PICL: 物理信息对比学习用于偏微分方程

    PICL: Physics Informed Contrastive Learning for Partial Differential Equations

    [https://arxiv.org/abs/2401.16327](https://arxiv.org/abs/2401.16327)

    这项工作开发了一种使用广义对比损失的对比预训练框架，通过利用物理信息改善了神经算子在多个偏微分方程中的泛化能力。

    

    最近，神经算子作为偏微分方程（PDE）替代模型逐渐受到关注。学习解决方案函数而不是函数本身已被证明是一种强大的方法，可快速准确地求解复杂的PDE。尽管在广泛的代理建模任务中对神经算子的性能进行了许多研究，但这些工作通常是逐个方程评估性能。在本研究中，我们开发了一种新颖的对比预训练框架，利用广义对比损失，可以同时改善神经算子在多个控制方程中的泛化能力。控制方程系数用于衡量系统之间的真实相似性。物理信息系统演化和潜在空间模型输出的结合被锚定到输入数据中，并用于我们的距离函数。我们发现，物理信息对比预训练可以提高傅立叶神经算子的准确性和泛化能力。

    Neural operators have recently grown in popularity as Partial Differential Equation (PDEs) surrogate models. Learning solution functionals, rather than functions, has proven to be a powerful approach to calculate fast, accurate solutions to complex PDEs. While much work has been done evaluating neural operator performance on a wide variety of surrogate modeling tasks, these works normally evaluate performance on a single equation at a time. In this work, we develop a novel contrastive pretraining framework utilizing Generalized Contrastive Loss that improves neural operator generalization across multiple governing equations simultaneously. Governing equation coefficients are used to measure ground-truth similarity between systems. A combination of physics-informed system evolution and latent-space model output are anchored to input data and used in our distance function. We find that physics-informed contrastive pretraining improves both accuracy and generalization for the Fourier Neur
    
[^2]: 自适应联合分布学习

    Adaptive joint distribution learning. (arXiv:2110.04829v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2110.04829](http://arxiv.org/abs/2110.04829)

    该论文提出了一种自适应联合分布学习的框架，可以从大量数据点中估计低维、归一化和正的Radon-Nikodym导数模型，并在不同学习问题上取得了良好的结果。

    

    我们开发了一个新的框架，用于将联合概率分布嵌入张量积再生核希尔伯特空间（RKHS）中。我们的框架可以容纳一个低维、归一化和正的Radon-Nikodym导数模型，该模型可以从多达数百万个数据点的样本大小中进行估计，减轻了RKHS建模的固有限制。我们的方法自然产生了定义良好的归一化和正的条件分布。嵌入计算速度快且适用于从预测到分类的各种学习问题。我们的理论结果得到了有益的数值结果的支持。

    We develop a new framework for embedding joint probability distributions in tensor product reproducing kernel Hilbert spaces (RKHS). Our framework accommodates a low-dimensional, normalized and positive model of a Radon-Nikodym derivative, which we estimate from sample sizes of up to several million data points, alleviating the inherent limitations of RKHS modeling. Well-defined normalized and positive conditional distributions are natural by-products to our approach. The embedding is fast to compute and accommodates learning problems ranging from prediction to classification. Our theoretical findings are supplemented by favorable numerical results.
    

