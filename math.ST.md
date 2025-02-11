# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Independent Samples Along the Langevin Diffusion and the Unadjusted Langevin Algorithm](https://arxiv.org/abs/2402.17067) | 在该论文中，我们研究了朗之凡扩散和未调整朗之凡算法中随机变量独立化速率的收敛性，证明了在目标函数强对数凹和平滑的情况下，互信息会以指数速率收敛于$0$。 |
| [^2] | [Understanding Generalization in the Interpolation Regime using the Rate Function.](http://arxiv.org/abs/2306.10947) | 本文利用大偏差理论，提出一种基于函数的平滑模型特征描述方法，解释了为什么一些插值器有很好的泛化能力以及现代学习技术为什么能够找到它们。 |

# 详细

[^1]: 关于朗之凡扩散和未调整朗之凡算法中独立样本的研究

    On Independent Samples Along the Langevin Diffusion and the Unadjusted Langevin Algorithm

    [https://arxiv.org/abs/2402.17067](https://arxiv.org/abs/2402.17067)

    在该论文中，我们研究了朗之凡扩散和未调整朗之凡算法中随机变量独立化速率的收敛性，证明了在目标函数强对数凹和平滑的情况下，互信息会以指数速率收敛于$0$。

    

    我们研究了马尔可夫链中初始和当前随机变量独立化的速率，重点关注连续时间中的朗之凡扩散和离散时间中的未调整朗之凡算法（ULA）。我们通过它们的互信息度量随机变量之间的依赖关系。对于朗之凡扩散，我们展示了当目标函数强对数凹时，互信息以指数速率收敛于$0$，当目标函数弱对数凹时，以多项式速率收敛。这些速率类似于在类似条件下朗之凡扩散的混合时间。对于ULA，我们展示了当目标函数强对数凹且光滑时，互信息以指数速率收敛于$0$。我们通过发展这些马尔可夫链的互信息版本的混合时间分析来证明我们的结果。我们还提供了基于朗之凡扩散的强数据处理不等式的替代证明。

    arXiv:2402.17067v1 Announce Type: cross  Abstract: We study the rate at which the initial and current random variables become independent along a Markov chain, focusing on the Langevin diffusion in continuous time and the Unadjusted Langevin Algorithm (ULA) in discrete time. We measure the dependence between random variables via their mutual information. For the Langevin diffusion, we show the mutual information converges to $0$ exponentially fast when the target is strongly log-concave, and at a polynomial rate when the target is weakly log-concave. These rates are analogous to the mixing time of the Langevin diffusion under similar assumptions. For the ULA, we show the mutual information converges to $0$ exponentially fast when the target is strongly log-concave and smooth. We prove our results by developing the mutual version of the mixing time analyses of these Markov chains. We also provide alternative proofs based on strong data processing inequalities for the Langevin diffusion 
    
[^2]: 使用速率函数理解插值区间的泛化

    Understanding Generalization in the Interpolation Regime using the Rate Function. (arXiv:2306.10947v1 [cs.LG])

    [http://arxiv.org/abs/2306.10947](http://arxiv.org/abs/2306.10947)

    本文利用大偏差理论，提出一种基于函数的平滑模型特征描述方法，解释了为什么一些插值器有很好的泛化能力以及现代学习技术为什么能够找到它们。

    

    本文基于大偏差理论的基本原理，提出了一种模型平滑度的新特征描述方法。与以往的工作不同，以往的工作通常用实数值（如权重范数）来表征模型的平滑度，我们表明可以用简单的实值函数来描述平滑度。基于模型平滑度的这一概念，我们提出了一个统一的理论解释，为什么一些插值器表现出非常好的泛化能力，以及为什么广泛使用的现代学习技术（如随机梯度下降，$\ell_2$-规范化，数据增强，不变的架构和超参数化）能够找到它们。我们得出的结论是，所有这些方法都提供了互补的过程，这些过程使优化器偏向于更平滑的插值器，而根据这种理论分析，更平滑的插值器是具有更好的泛化误差的插值器。

    In this paper, we present a novel characterization of the smoothness of a model based on basic principles of Large Deviation Theory. In contrast to prior work, where the smoothness of a model is normally characterized by a real value (e.g., the weights' norm), we show that smoothness can be described by a simple real-valued function. Based on this concept of smoothness, we propose an unifying theoretical explanation of why some interpolators generalize remarkably well and why a wide range of modern learning techniques (i.e., stochastic gradient descent, $\ell_2$-norm regularization, data augmentation, invariant architectures, and overparameterization) are able to find them. The emergent conclusion is that all these methods provide complimentary procedures that bias the optimizer to smoother interpolators, which, according to this theoretical analysis, are the ones with better generalization error.
    

