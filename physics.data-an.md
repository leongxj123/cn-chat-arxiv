# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Characterizing the load profile in power grids by Koopman mode decomposition of interconnected dynamics.](http://arxiv.org/abs/2304.07832) | 本论文介绍了一种基于Koopman算子的负载动态分解方法，能够编码电网负载动态的丰富特征，提高预测精度，同时提供有意义的解释。 |

# 详细

[^1]: 基于Koopman模态分解的电网负载分析

    Characterizing the load profile in power grids by Koopman mode decomposition of interconnected dynamics. (arXiv:2304.07832v1 [cs.LG])

    [http://arxiv.org/abs/2304.07832](http://arxiv.org/abs/2304.07832)

    本论文介绍了一种基于Koopman算子的负载动态分解方法，能够编码电网负载动态的丰富特征，提高预测精度，同时提供有意义的解释。

    

    电力负载预测对于有效管理和优化电网至关重要。本文介绍了一种可解释的机器学习方法，利用算子理论框架内的数据驱动方法识别负载动态。我们使用Koopman算子来表示负载数据，该算子固有于底层动态。通过计算相应的特征函数，我们将负载动态分解为相干的时空模式，这些模式是动态的最强特征。每个模式根据其单一频率独立演化，基于线性动力学的可预测性。我们强调，负载动态是基于固有于动态的相干的时空模式构建的，能够在多个时间尺度上编码丰富的动态特征。这些特征与电网的物理特征（如季节性和小时模式）有关。我们的方法实现了最先进的预测准确性，同时提供了对底层动态的有意义的解释。

    Electricity load forecasting is crucial for effectively managing and optimizing power grids. Over the past few decades, various statistical and deep learning approaches have been used to develop load forecasting models. This paper presents an interpretable machine learning approach that identifies load dynamics using data-driven methods within an operator-theoretic framework. We represent the load data using the Koopman operator, which is inherent to the underlying dynamics. By computing the corresponding eigenfunctions, we decompose the load dynamics into coherent spatiotemporal patterns that are the most robust features of the dynamics. Each pattern evolves independently according to its single frequency, making its predictability based on linear dynamics. We emphasize that the load dynamics are constructed based on coherent spatiotemporal patterns that are intrinsic to the dynamics and are capable of encoding rich dynamical features at multiple time scales. These features are relate
    

