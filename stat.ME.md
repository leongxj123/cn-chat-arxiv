# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TWIN-GPT: Digital Twins for Clinical Trials via Large Language Model](https://arxiv.org/abs/2404.01273) | 提出了基于大语言模型的数字孪生体TWIN-GPT，用于支持临床试验结果预测。 |
| [^2] | [A flexible Bayesian g-formula for causal survival analyses with time-dependent confounding](https://arxiv.org/abs/2402.02306) | 本文提出了一种更灵活的贝叶斯g形式估计器，用于具有时变混杂的因果生存分析。它采用贝叶斯附加回归树来模拟时变生成组件，并引入了纵向平衡分数以降低模型错误规范引起的偏差。 |
| [^3] | [Entrywise Inference for Causal Panel Data: A Simple and Instance-Optimal Approach.](http://arxiv.org/abs/2401.13665) | 本研究提出了一种在面板数据中进行因果推断的简单且最佳化的方法，通过简单的矩阵代数和奇异值分解来实现高效计算。通过导出非渐近界限，我们证明了逐项误差与高斯变量的适当缩放具有接近性。同时，我们还开发了一个数据驱动程序，用于构建具有预先指定覆盖保证的逐项置信区间。 |
| [^4] | [Stationary Kernels and Gaussian Processes on Lie Groups and their Homogeneous Spaces II: non-compact symmetric spaces.](http://arxiv.org/abs/2301.13088) | 本文开发了构建非欧几里得空间上静止高斯过程的实用技术，能够对定义在这些空间上的先验和后验高斯过程进行实际采样和计算协方差核。 |

# 详细

[^1]: TWIN-GPT: 基于大语言模型的临床试验数字孪生体

    TWIN-GPT: Digital Twins for Clinical Trials via Large Language Model

    [https://arxiv.org/abs/2404.01273](https://arxiv.org/abs/2404.01273)

    提出了基于大语言模型的数字孪生体TWIN-GPT，用于支持临床试验结果预测。

    

    最近，对虚拟临床试验产生了日益增长的兴趣，这些试验模拟了现实世界情境，有望显著增强患者安全性，加快开发速度，降低成本，并为医疗领域的更广泛科学知识贡献力量。本文提出了一种基于大语言模型的数字孪生体TWIN-GPT，用于支持临床试验结果预测。

    arXiv:2404.01273v1 Announce Type: cross  Abstract: Recently, there has been a burgeoning interest in virtual clinical trials, which simulate real-world scenarios and hold the potential to significantly enhance patient safety, expedite development, reduce costs, and contribute to the broader scientific knowledge in healthcare. Existing research often focuses on leveraging electronic health records (EHRs) to support clinical trial outcome prediction. Yet, trained with limited clinical trial outcome data, existing approaches frequently struggle to perform accurate predictions. Some research has attempted to generate EHRs to augment model development but has fallen short in personalizing the generation for individual patient profiles. Recently, the emergence of large language models has illuminated new possibilities, as their embedded comprehensive clinical knowledge has proven beneficial in addressing medical issues. In this paper, we propose a large language model-based digital twin crea
    
[^2]: 弹性贝叶斯g形式在具有时变混杂的因果生存分析中的应用

    A flexible Bayesian g-formula for causal survival analyses with time-dependent confounding

    [https://arxiv.org/abs/2402.02306](https://arxiv.org/abs/2402.02306)

    本文提出了一种更灵活的贝叶斯g形式估计器，用于具有时变混杂的因果生存分析。它采用贝叶斯附加回归树来模拟时变生成组件，并引入了纵向平衡分数以降低模型错误规范引起的偏差。

    

    在具有时间至事件结果的纵向观察性研究中，因果分析的常见目标是在研究群体中估计在假设干预情景下的因果生存曲线。g形式是这种分析的一个特别有用的工具。为了增强传统的参数化g形式方法，我们开发了一种更灵活的贝叶斯g形式估计器。该估计器同时支持纵向预测和因果推断。它在模拟时变生成组件的建模中引入了贝叶斯附加回归树，旨在减轻由于模型错误规范造成的偏差。具体而言，我们引入了一类更通用的离散生存数据g形式。这些公式可以引入纵向平衡分数，这在处理越来越多的时变混杂因素时是一种有效的降维方法。

    In longitudinal observational studies with a time-to-event outcome, a common objective in causal analysis is to estimate the causal survival curve under hypothetical intervention scenarios within the study cohort. The g-formula is a particularly useful tool for this analysis. To enhance the traditional parametric g-formula approach, we developed a more adaptable Bayesian g-formula estimator. This estimator facilitates both longitudinal predictive and causal inference. It incorporates Bayesian additive regression trees in the modeling of the time-evolving generative components, aiming to mitigate bias due to model misspecification. Specifically, we introduce a more general class of g-formulas for discrete survival data. These formulas can incorporate the longitudinal balancing scores, which serve as an effective method for dimension reduction and are vital when dealing with an expanding array of time-varying confounders. The minimum sufficient formulation of these longitudinal balancing
    
[^3]: 《面板数据因果推断的逐项推理方法：一种简单且最佳化的方法》

    Entrywise Inference for Causal Panel Data: A Simple and Instance-Optimal Approach. (arXiv:2401.13665v1 [math.ST])

    [http://arxiv.org/abs/2401.13665](http://arxiv.org/abs/2401.13665)

    本研究提出了一种在面板数据中进行因果推断的简单且最佳化的方法，通过简单的矩阵代数和奇异值分解来实现高效计算。通过导出非渐近界限，我们证明了逐项误差与高斯变量的适当缩放具有接近性。同时，我们还开发了一个数据驱动程序，用于构建具有预先指定覆盖保证的逐项置信区间。

    

    在分阶段采用的面板数据中的因果推断中，目标是估计和推导出潜在结果和处理效应的置信区间。我们提出了一种计算效率高的程序，仅涉及简单的矩阵代数和奇异值分解。我们导出了逐项误差的非渐近界限，证明其接近于适当缩放的高斯变量。尽管我们的程序简单，但却是局部最佳化的，因为我们的理论缩放与通过贝叶斯Cram\'{e}r-Rao论证得出的局部实例下界相匹配。利用我们的见解，我们开发了一种数据驱动的程序，用于构建具有预先指定覆盖保证的逐项置信区间。我们的分析基于对矩阵去噪模型应用SVD算法的一般推理工具箱，这可能具有独立的兴趣。

    In causal inference with panel data under staggered adoption, the goal is to estimate and derive confidence intervals for potential outcomes and treatment effects. We propose a computationally efficient procedure, involving only simple matrix algebra and singular value decomposition. We derive non-asymptotic bounds on the entrywise error, establishing its proximity to a suitably scaled Gaussian variable. Despite its simplicity, our procedure turns out to be instance-optimal, in that our theoretical scaling matches a local instance-wise lower bound derived via a Bayesian Cram\'{e}r-Rao argument. Using our insights, we develop a data-driven procedure for constructing entrywise confidence intervals with pre-specified coverage guarantees. Our analysis is based on a general inferential toolbox for the SVD algorithm applied to the matrix denoising model, which might be of independent interest.
    
[^4]: Lie 群和它们的齐次空间上的静止核和高斯过程 II：非紧对称空间

    Stationary Kernels and Gaussian Processes on Lie Groups and their Homogeneous Spaces II: non-compact symmetric spaces. (arXiv:2301.13088v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2301.13088](http://arxiv.org/abs/2301.13088)

    本文开发了构建非欧几里得空间上静止高斯过程的实用技术，能够对定义在这些空间上的先验和后验高斯过程进行实际采样和计算协方差核。

    

    高斯过程是机器学习中最重要的时空模型之一，它可以编码有关建模函数的先验信息，并可用于精确或近似贝叶斯学习。在许多应用中，特别是在物理科学和工程领域，以及地质统计学和神经科学等领域，对对称性的不变性是可以考虑的最基本形式之一。高斯过程协方差对这些对称性的不变性引发了对这些空间的平稳性概念的最自然的推广。在这项工作中，我们开发了建立静止高斯过程的构造性和实用技术，用于在对称性背景下出现的非欧几里得空间的非常大的类。我们的技术使得能够（i）计算协方差核和（ii）从这些空间上定义的先验和后验高斯过程中实际地进行采样。

    Gaussian processes are arguably the most important class of spatiotemporal models within machine learning. They encode prior information about the modeled function and can be used for exact or approximate Bayesian learning. In many applications, particularly in physical sciences and engineering, but also in areas such as geostatistics and neuroscience, invariance to symmetries is one of the most fundamental forms of prior information one can consider. The invariance of a Gaussian process' covariance to such symmetries gives rise to the most natural generalization of the concept of stationarity to such spaces. In this work, we develop constructive and practical techniques for building stationary Gaussian processes on a very large class of non-Euclidean spaces arising in the context of symmetries. Our techniques make it possible to (i) calculate covariance kernels and (ii) sample from prior and posterior Gaussian processes defined on such spaces, both in a practical manner. This work is 
    

