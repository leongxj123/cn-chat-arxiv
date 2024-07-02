# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Entrywise Inference for Causal Panel Data: A Simple and Instance-Optimal Approach.](http://arxiv.org/abs/2401.13665) | 本研究提出了一种在面板数据中进行因果推断的简单且最佳化的方法，通过简单的矩阵代数和奇异值分解来实现高效计算。通过导出非渐近界限，我们证明了逐项误差与高斯变量的适当缩放具有接近性。同时，我们还开发了一个数据驱动程序，用于构建具有预先指定覆盖保证的逐项置信区间。 |
| [^2] | [Stationary Kernels and Gaussian Processes on Lie Groups and their Homogeneous Spaces II: non-compact symmetric spaces.](http://arxiv.org/abs/2301.13088) | 本文开发了构建非欧几里得空间上静止高斯过程的实用技术，能够对定义在这些空间上的先验和后验高斯过程进行实际采样和计算协方差核。 |

# 详细

[^1]: 《面板数据因果推断的逐项推理方法：一种简单且最佳化的方法》

    Entrywise Inference for Causal Panel Data: A Simple and Instance-Optimal Approach. (arXiv:2401.13665v1 [math.ST])

    [http://arxiv.org/abs/2401.13665](http://arxiv.org/abs/2401.13665)

    本研究提出了一种在面板数据中进行因果推断的简单且最佳化的方法，通过简单的矩阵代数和奇异值分解来实现高效计算。通过导出非渐近界限，我们证明了逐项误差与高斯变量的适当缩放具有接近性。同时，我们还开发了一个数据驱动程序，用于构建具有预先指定覆盖保证的逐项置信区间。

    

    在分阶段采用的面板数据中的因果推断中，目标是估计和推导出潜在结果和处理效应的置信区间。我们提出了一种计算效率高的程序，仅涉及简单的矩阵代数和奇异值分解。我们导出了逐项误差的非渐近界限，证明其接近于适当缩放的高斯变量。尽管我们的程序简单，但却是局部最佳化的，因为我们的理论缩放与通过贝叶斯Cram\'{e}r-Rao论证得出的局部实例下界相匹配。利用我们的见解，我们开发了一种数据驱动的程序，用于构建具有预先指定覆盖保证的逐项置信区间。我们的分析基于对矩阵去噪模型应用SVD算法的一般推理工具箱，这可能具有独立的兴趣。

    In causal inference with panel data under staggered adoption, the goal is to estimate and derive confidence intervals for potential outcomes and treatment effects. We propose a computationally efficient procedure, involving only simple matrix algebra and singular value decomposition. We derive non-asymptotic bounds on the entrywise error, establishing its proximity to a suitably scaled Gaussian variable. Despite its simplicity, our procedure turns out to be instance-optimal, in that our theoretical scaling matches a local instance-wise lower bound derived via a Bayesian Cram\'{e}r-Rao argument. Using our insights, we develop a data-driven procedure for constructing entrywise confidence intervals with pre-specified coverage guarantees. Our analysis is based on a general inferential toolbox for the SVD algorithm applied to the matrix denoising model, which might be of independent interest.
    
[^2]: Lie 群和它们的齐次空间上的静止核和高斯过程 II：非紧对称空间

    Stationary Kernels and Gaussian Processes on Lie Groups and their Homogeneous Spaces II: non-compact symmetric spaces. (arXiv:2301.13088v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2301.13088](http://arxiv.org/abs/2301.13088)

    本文开发了构建非欧几里得空间上静止高斯过程的实用技术，能够对定义在这些空间上的先验和后验高斯过程进行实际采样和计算协方差核。

    

    高斯过程是机器学习中最重要的时空模型之一，它可以编码有关建模函数的先验信息，并可用于精确或近似贝叶斯学习。在许多应用中，特别是在物理科学和工程领域，以及地质统计学和神经科学等领域，对对称性的不变性是可以考虑的最基本形式之一。高斯过程协方差对这些对称性的不变性引发了对这些空间的平稳性概念的最自然的推广。在这项工作中，我们开发了建立静止高斯过程的构造性和实用技术，用于在对称性背景下出现的非欧几里得空间的非常大的类。我们的技术使得能够（i）计算协方差核和（ii）从这些空间上定义的先验和后验高斯过程中实际地进行采样。

    Gaussian processes are arguably the most important class of spatiotemporal models within machine learning. They encode prior information about the modeled function and can be used for exact or approximate Bayesian learning. In many applications, particularly in physical sciences and engineering, but also in areas such as geostatistics and neuroscience, invariance to symmetries is one of the most fundamental forms of prior information one can consider. The invariance of a Gaussian process' covariance to such symmetries gives rise to the most natural generalization of the concept of stationarity to such spaces. In this work, we develop constructive and practical techniques for building stationary Gaussian processes on a very large class of non-Euclidean spaces arising in the context of symmetries. Our techniques make it possible to (i) calculate covariance kernels and (ii) sample from prior and posterior Gaussian processes defined on such spaces, both in a practical manner. This work is 
    

