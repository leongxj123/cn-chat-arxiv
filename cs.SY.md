# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-Source Domain Adaptation for Cross-Domain Fault Diagnosis of Chemical Processes.](http://arxiv.org/abs/2308.11247) | 本文在化学过程的交叉领域故障诊断中，对单源和多源无监督领域适应算法进行了广泛比较。研究结果表明，即使没有进行适应，使用多个领域进行训练也具有积极影响。 |
| [^2] | [Meta-Learning Operators to Optimality from Multi-Task Non-IID Data.](http://arxiv.org/abs/2308.04428) | 本文提出了从多任务非独立同分布数据中恢复线性操作符的方法，并发现现有的各向同性无关的元学习方法会对表示更新造成偏差，限制了表示学习的样本复杂性。为此，引入了去偏差和特征白化的适应方法。 |
| [^3] | [Exact Characterization of the Convex Hulls of Reachable Sets.](http://arxiv.org/abs/2303.17674) | 本文精确地刻画了具有有界扰动的非线性系统的可达集的凸包为一阶常微分方程的解的凸包，提出了一种低成本、高精度的估计算法，可用于过逼近可达集。 |

# 详细

[^1]: 多源领域适应用于化学过程交叉领域故障诊断

    Multi-Source Domain Adaptation for Cross-Domain Fault Diagnosis of Chemical Processes. (arXiv:2308.11247v1 [cs.LG])

    [http://arxiv.org/abs/2308.11247](http://arxiv.org/abs/2308.11247)

    本文在化学过程的交叉领域故障诊断中，对单源和多源无监督领域适应算法进行了广泛比较。研究结果表明，即使没有进行适应，使用多个领域进行训练也具有积极影响。

    

    故障诊断是过程监视中的重要组成部分。机器学习的故障诊断系统基于传感器数据预测故障类型。然而，这些模型对数据分布的变化敏感，这些变化可能由于监测过程中的变化，如操作模式的改变，导致跨领域故障诊断的情况。本文在化学工业中广泛使用的田纳西-伊斯曼过程的背景下，提供了单源和多源无监督领域适应算法在交叉领域故障诊断中的广泛比较。研究结果表明，即使没有进行适应，使用多个领域进行训练也具有积极影响。因此，多源无监督领域适应的基准模型相对于单源无监督领域适应的基准模型有所改进。

    Fault diagnosis is an essential component in process supervision. Indeed, it determines which kind of fault has occurred, given that it has been previously detected, allowing for appropriate intervention. Automatic fault diagnosis systems use machine learning for predicting the fault type from sensor readings. Nonetheless, these models are sensible to changes in the data distributions, which may be caused by changes in the monitored process, such as changes in the mode of operation. This scenario is known as Cross-Domain Fault Diagnosis (CDFD). We provide an extensive comparison of single and multi-source unsupervised domain adaptation (SSDA and MSDA respectively) algorithms for CDFD. We study these methods in the context of the Tennessee-Eastmann Process, a widely used benchmark in the chemical industry. We show that using multiple domains during training has a positive effect, even when no adaptation is employed. As such, the MSDA baseline improves over the SSDA baseline classificati
    
[^2]: 从多任务非独立同分布数据中元学习操作符到最优性

    Meta-Learning Operators to Optimality from Multi-Task Non-IID Data. (arXiv:2308.04428v1 [stat.ML])

    [http://arxiv.org/abs/2308.04428](http://arxiv.org/abs/2308.04428)

    本文提出了从多任务非独立同分布数据中恢复线性操作符的方法，并发现现有的各向同性无关的元学习方法会对表示更新造成偏差，限制了表示学习的样本复杂性。为此，引入了去偏差和特征白化的适应方法。

    

    机器学习中最近取得进展的一个强大概念是从异构来源或任务的数据中提取共同特征。直观地说，将所有数据用于学习共同的表示函数，既有助于计算效率，又有助于统计泛化，因为它可以减少要在给定任务上进行微调的参数数量。为了在理论上做出这些优点的根源，我们提出了从噪声向量测量$y = Mx + w$中回复线性操作符$M$的一般模型。其中，协变量$x$既可以是非独立同分布的，也可以是非各向同性的。我们证明了现有的各向同性无关的元学习方法会对表示更新造成偏差，这导致噪声项的缩放不再有利于源任务数量。这反过来会导致表示学习的样本复杂性受到单任务数据规模的限制。我们引入了一种方法，称为去偏差和特征白化。

    A powerful concept behind much of the recent progress in machine learning is the extraction of common features across data from heterogeneous sources or tasks. Intuitively, using all of one's data to learn a common representation function benefits both computational effort and statistical generalization by leaving a smaller number of parameters to fine-tune on a given task. Toward theoretically grounding these merits, we propose a general setting of recovering linear operators $M$ from noisy vector measurements $y = Mx + w$, where the covariates $x$ may be both non-i.i.d. and non-isotropic. We demonstrate that existing isotropy-agnostic meta-learning approaches incur biases on the representation update, which causes the scaling of the noise terms to lose favorable dependence on the number of source tasks. This in turn can cause the sample complexity of representation learning to be bottlenecked by the single-task data size. We introduce an adaptation, $\texttt{De-bias & Feature-Whiten}
    
[^3]: 可达集的凸包的精确刻画

    Exact Characterization of the Convex Hulls of Reachable Sets. (arXiv:2303.17674v1 [math.OC])

    [http://arxiv.org/abs/2303.17674](http://arxiv.org/abs/2303.17674)

    本文精确地刻画了具有有界扰动的非线性系统的可达集的凸包为一阶常微分方程的解的凸包，提出了一种低成本、高精度的估计算法，可用于过逼近可达集。

    

    本文研究了具有有界扰动的非线性系统的可达集的凸包。可达集在控制中起着至关重要的作用，但计算起来仍然非常具有挑战性，现有的过逼近工具往往过于保守或计算代价高昂。本文精确地刻画了可达集的凸包，将其表示成一阶常微分方程的解的凸包，这个有限维的刻画开启了一种紧密的估计算法，可用于过逼近可达集，且成本比现有方法更低、更精准。本文还提出了神经反馈环分析和鲁棒模型预测控制的应用。

    We study the convex hulls of reachable sets of nonlinear systems with bounded disturbances. Reachable sets play a critical role in control, but remain notoriously challenging to compute, and existing over-approximation tools tend to be conservative or computationally expensive. In this work, we exactly characterize the convex hulls of reachable sets as the convex hulls of solutions of an ordinary differential equation from all possible initial values of the disturbances. This finite-dimensional characterization unlocks a tight estimation algorithm to over-approximate reachable sets that is significantly faster and more accurate than existing methods. We present applications to neural feedback loop analysis and robust model predictive control.
    

