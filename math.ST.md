# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Model-based causal feature selection for general response types.](http://arxiv.org/abs/2309.12833) | 本研究基于模型提出了一种通用响应类型的因果特征选择方法，该方法利用不变性假设从异质环境的数据中输出一部分因果特征的子集，适用于一般的加性噪声模型和非参数设置，解决了非参数条件独立性测试低功率的问题。 |

# 详细

[^1]: 基于模型的通用响应类型因果特征选择

    Model-based causal feature selection for general response types. (arXiv:2309.12833v1 [stat.ME])

    [http://arxiv.org/abs/2309.12833](http://arxiv.org/abs/2309.12833)

    本研究基于模型提出了一种通用响应类型的因果特征选择方法，该方法利用不变性假设从异质环境的数据中输出一部分因果特征的子集，适用于一般的加性噪声模型和非参数设置，解决了非参数条件独立性测试低功率的问题。

    

    从观测数据中发现因果关系是一项基本而具有挑战性的任务。在某些应用中，仅学习给定响应变量的因果特征可能已经足够，而不是学习整个潜在的因果结构。不变因果预测（ICP）是一种用于因果特征选择的方法，需要来自异质环境的数据。ICP假设从直接原因生成响应的机制在所有环境中都相同，并利用这种不变性输出一部分因果特征的子集。ICP的框架已经扩展到一般的加性噪声模型和非参数设置，使用条件独立性测试。然而，非参数条件独立性测试经常受到低功率（或较差的类型I错误控制）的困扰，并且上述参数模型不适用于响应不是在连续刻度上测量的应用情况，而是反映了分类信息的情况。

    Discovering causal relationships from observational data is a fundamental yet challenging task. In some applications, it may suffice to learn the causal features of a given response variable, instead of learning the entire underlying causal structure. Invariant causal prediction (ICP, Peters et al., 2016) is a method for causal feature selection which requires data from heterogeneous settings. ICP assumes that the mechanism for generating the response from its direct causes is the same in all settings and exploits this invariance to output a subset of the causal features. The framework of ICP has been extended to general additive noise models and to nonparametric settings using conditional independence testing. However, nonparametric conditional independence testing often suffers from low power (or poor type I error control) and the aforementioned parametric models are not suitable for applications in which the response is not measured on a continuous scale, but rather reflects categor
    

