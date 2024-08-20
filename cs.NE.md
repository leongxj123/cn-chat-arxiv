# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Chain of Compression: A Systematic Approach to Combinationally Compress Convolutional Neural Networks](https://arxiv.org/abs/2403.17447) | 提出了一种名为“压缩链”的系统化方法，通过结合量化、剪枝、提前退出和知识蒸馏等常见技术，实现对卷积神经网络的压缩。 |
| [^2] | [Auto-weighted Bayesian Physics-Informed Neural Networks and robust estimations for multitask inverse problems in pore-scale imaging of dissolution.](http://arxiv.org/abs/2308.12864) | 本文介绍了一种新的数据同化策略，可可靠地处理包含不确定度量化的孔隙尺度反应反问题。该方法结合了数据驱动和物理建模，确保了孔隙尺度模型的可靠校准。 |
| [^3] | [Metaheuristic Algorithms in Artificial Intelligence with Applications to Bioinformatics, Biostatistics, Ecology and, the Manufacturing Industries.](http://arxiv.org/abs/2308.10875) | 这篇论文介绍了受自然启发的元启发式算法在人工智能中的重要性和应用，并提出了一种新算法CSO-MA，通过多个优化问题的应用展示了其灵活性和优越性能。 |
| [^4] | [Patch of Invisibility: Naturalistic Black-Box Adversarial Attacks on Object Detectors.](http://arxiv.org/abs/2303.04238) | 本文提出了一种基于GAN的无梯度物理对抗攻击方法，用于生成自然的对抗补丁，攻击物体检测器，具有实际应用价值。 |

# 详细

[^1]: 压缩链：一种系统化的组合压缩卷积神经网络方法

    Chain of Compression: A Systematic Approach to Combinationally Compress Convolutional Neural Networks

    [https://arxiv.org/abs/2403.17447](https://arxiv.org/abs/2403.17447)

    提出了一种名为“压缩链”的系统化方法，通过结合量化、剪枝、提前退出和知识蒸馏等常见技术，实现对卷积神经网络的压缩。

    

    卷积神经网络（CNNs）已经取得了显著的流行，但它们在计算和存储方面的密集性给资源有限的计算系统带来了挑战，尤其是在需要实时性能的情况下。为了减轻负担，模型压缩已经成为一个重要的研究重点。许多方法，如量化、剪枝、提前退出和知识蒸馏已经证明了减少神经网络中冗余的效果。通过进一步的研究，可以明显看出，每种方法都利用了其独特的特性来压缩神经网络，并且当它们结合在一起时也可以展现出互补的行为。为了探究这些相互作用，并从互补特性中获益，我们提出了压缩链，它在组合序列上操作，应用这些常见技术来压缩神经网络。

    arXiv:2403.17447v1 Announce Type: new  Abstract: Convolutional neural networks (CNNs) have achieved significant popularity, but their computational and memory intensity poses challenges for resource-constrained computing systems, particularly with the prerequisite of real-time performance. To release this burden, model compression has become an important research focus. Many approaches like quantization, pruning, early exit, and knowledge distillation have demonstrated the effect of reducing redundancy in neural networks. Upon closer examination, it becomes apparent that each approach capitalizes on its unique features to compress the neural network, and they can also exhibit complementary behavior when combined. To explore the interactions and reap the benefits from the complementary features, we propose the Chain of Compression, which works on the combinational sequence to apply these common techniques to compress the neural network. Validated on the image-based regression and classi
    
[^2]: 自动加权的贝叶斯物理信息神经网络和鲁棒估计在孔隙尺度溶解图像的多任务反问题中的应用

    Auto-weighted Bayesian Physics-Informed Neural Networks and robust estimations for multitask inverse problems in pore-scale imaging of dissolution. (arXiv:2308.12864v1 [cs.LG])

    [http://arxiv.org/abs/2308.12864](http://arxiv.org/abs/2308.12864)

    本文介绍了一种新的数据同化策略，可可靠地处理包含不确定度量化的孔隙尺度反应反问题。该方法结合了数据驱动和物理建模，确保了孔隙尺度模型的可靠校准。

    

    在这篇文章中，我们提出了一种新的数据同化策略，并展示了这种方法可以使我们能够可靠地处理包含不确定度量化的反应反问题。孔隙尺度的反应流动建模为研究宏观性质在动态过程中的演变提供了宝贵的机会。然而，它们受到相关的X射线微计算高度分辨率成像 (X射线微CT) 过程中的成像限制的影响，导致了性质估计中的差异。动力学参数的评估也面临挑战，因为反应系数是关键参数，其数值范围很广。我们解决了这两个问题，并通过将不确定度量化集成到工作流程中，确保了孔隙尺度模型的可靠校准。当前的方法基于反应反问题的多任务公式，将数据驱动和物理建模相结合。

    In this article, we present a novel data assimilation strategy in pore-scale imaging and demonstrate that this makes it possible to robustly address reactive inverse problems incorporating Uncertainty Quantification (UQ). Pore-scale modeling of reactive flow offers a valuable opportunity to investigate the evolution of macro-scale properties subject to dynamic processes. Yet, they suffer from imaging limitations arising from the associated X-ray microtomography (X-ray microCT) process, which induces discrepancies in the properties estimates. Assessment of the kinetic parameters also raises challenges, as reactive coefficients are critical parameters that can cover a wide range of values. We account for these two issues and ensure reliable calibration of pore-scale modeling, based on dynamical microCT images, by integrating uncertainty quantification in the workflow.  The present method is based on a multitasking formulation of reactive inverse problems combining data-driven and physics
    
[^3]: 人工智能中的元启发式算法及其在生物信息学、生物统计学、生态学和制造业中的应用

    Metaheuristic Algorithms in Artificial Intelligence with Applications to Bioinformatics, Biostatistics, Ecology and, the Manufacturing Industries. (arXiv:2308.10875v2 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2308.10875](http://arxiv.org/abs/2308.10875)

    这篇论文介绍了受自然启发的元启发式算法在人工智能中的重要性和应用，并提出了一种新算法CSO-MA，通过多个优化问题的应用展示了其灵活性和优越性能。

    

    受自然启发的元启发式算法是人工智能的重要组成部分，并在不同学科领域中应用于解决各种类型的挑战性优化问题。我们应用了一种新提出的受自然启发的元启发式算法，称为具有突变代理的竞争性群体优化器(CSO-MA)，并证明了它相对于竞争对手在统计科学中各种优化问题上的灵活性和超越性能。特别是，我们展示了该算法高效且可以整合各种成本结构或多个用户指定的非线性约束。我们的应用包括(i)在生物信息学中通过单细胞广义趋势模型找到参数的最大似然估计以研究伪时态，(ii) 估计教育研究中常用的Rasch模型的参数，(iii) 在马尔可夫更新模型中为Cox回归找到M-估计，(iv) 矩阵补全以填补两个连连不通图中的缺失值。

    Nature-inspired metaheuristic algorithms are important components of artificial intelligence, and are increasingly used across disciplines to tackle various types of challenging optimization problems. We apply a newly proposed nature-inspired metaheuristic algorithm called competitive swarm optimizer with mutated agents (CSO-MA) and demonstrate its flexibility and out-performance relative to its competitors in a variety of optimization problems in the statistical sciences. In particular, we show the algorithm is efficient and can incorporate various cost structures or multiple user-specified nonlinear constraints. Our applications include (i) finding maximum likelihood estimates of parameters in a single cell generalized trend model to study pseudotime in bioinformatics, (ii) estimating parameters in a commonly used Rasch model in education research, (iii) finding M-estimates for a Cox regression in a Markov renewal model and (iv) matrix completion to impute missing values in a two com
    
[^4]: 区域隐形补丁：基于生成对抗网络的物理对抗攻击物体检测器

    Patch of Invisibility: Naturalistic Black-Box Adversarial Attacks on Object Detectors. (arXiv:2303.04238v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2303.04238](http://arxiv.org/abs/2303.04238)

    本文提出了一种基于GAN的无梯度物理对抗攻击方法，用于生成自然的对抗补丁，攻击物体检测器，具有实际应用价值。

    

    近年来，深度学习模型的对抗攻击越来越引起关注。这一领域的研究大多集中在基于梯度的技术，即所谓的白盒攻击，在其中攻击者可以访问目标模型的内部参数。然而，这种假设在实际世界中通常是不现实的。相对地，我们提出了一种在无需使用梯度的情况下，利用预先训练的生成对抗网络（GAN）的学习图像流形来生成自然的物理对抗补丁，用于物体检测器的攻击方法。我们展示了我们提出的方法在数字和物理层面上均可行。

    Adversarial attacks on deep-learning models have been receiving increased attention in recent years. Work in this area has mostly focused on gradient-based techniques, so-called white-box attacks, wherein the attacker has access to the targeted model's internal parameters; such an assumption is usually unrealistic in the real world. Some attacks additionally use the entire pixel space to fool a given model, which is neither practical nor physical (i.e., real-world). On the contrary, we propose herein a gradient-free method that uses the learned image manifold of a pretrained generative adversarial network (GAN) to generate naturalistic physical adversarial patches for object detectors. We show that our proposed method works both digitally and physically.
    

