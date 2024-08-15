# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Limit Order Book Dynamics and Order Size Modelling Using Compound Hawkes Process](https://arxiv.org/abs/2312.08927) | 本文提出了一种使用复合霍克斯进程建模限价单簿动态和订单尺寸的新方法，以校准分布抽取每个事件的订单尺寸，并在模型中保持正的价差。进一步地，我们根据时间条件模型参数支持经验观察，并使用改进的非参数方法校准霍克斯核函数和抑制性交叉激发核函数。 |
| [^2] | [Significance Bands for Local Projections.](http://arxiv.org/abs/2306.03073) | 本文表明，在局部投影分析中，应该使用显著性带来评估措施对结果的影响，而不是使用常用的置信度带。 |

# 详细

[^1]: 限价单簿动态与订单尺寸建模：复合霍克斯进程

    Limit Order Book Dynamics and Order Size Modelling Using Compound Hawkes Process

    [https://arxiv.org/abs/2312.08927](https://arxiv.org/abs/2312.08927)

    本文提出了一种使用复合霍克斯进程建模限价单簿动态和订单尺寸的新方法，以校准分布抽取每个事件的订单尺寸，并在模型中保持正的价差。进一步地，我们根据时间条件模型参数支持经验观察，并使用改进的非参数方法校准霍克斯核函数和抑制性交叉激发核函数。

    

    霍克斯进程已在文献中多种方式被用于模拟限价单簿动态，但往往仅关注事件间隔，而订单尺寸通常被假设为常数。我们提出了一种新颖的方法，使用复合霍克斯进程来模拟限价单簿，其中每个事件的订单尺寸来自校准分布。该方法以一种新颖的方式构建，使进程的价差始终保持正值。此外，我们根据时间条件模型参数以支持经验观察。我们使用改进的非参数方法来校准霍克斯核函数，并允许抑制性交叉激发核函数。我们展示了在纳斯达克交易所中一只股票的限价单簿上的结果和适度程度。

    Hawkes Process has been used to model Limit Order Book (LOB) dynamics in several ways in the literature however the focus has been limited to capturing the inter-event times while the order size is usually assumed to be constant. We propose a novel methodology of using Compound Hawkes Process for the LOB where each event has an order size sampled from a calibrated distribution. The process is formulated in a novel way such that the spread of the process always remains positive. Further, we condition the model parameters on time of day to support empirical observations. We make use of an enhanced non-parametric method to calibrate the Hawkes kernels and allow for inhibitory cross-excitation kernels. We showcase the results and quality of fits for an equity stock's LOB in the NASDAQ exchange.
    
[^2]: 局部投影的显著性带

    Significance Bands for Local Projections. (arXiv:2306.03073v1 [econ.EM])

    [http://arxiv.org/abs/2306.03073](http://arxiv.org/abs/2306.03073)

    本文表明，在局部投影分析中，应该使用显著性带来评估措施对结果的影响，而不是使用常用的置信度带。

    

    冲击反应函数描述了刺激或治疗后结果变量的动态演变。一个常见的兴趣假设是治疗是否影响了结果。我们表明，最好使用显著性带来评估这个假设，而不是依赖于通常显示的置信度带。在零假设下，我们展示了使用LM原则可以使用标准统计软件轻松构建显著性带，并且在图形化显示冲击反应时应当作为常规报告。

    An impulse response function describes the dynamic evolution of an outcome variable following a stimulus or treatment. A common hypothesis of interest is whether the treatment affects the outcome. We show that this hypothesis is best assessed using significance bands rather than relying on commonly displayed confidence bands. Under the null hypothesis, we show that significance bands are trivial to construct with standard statistical software using the LM principle, and should be reported as a matter of routine when displaying impulse responses graphically.
    

