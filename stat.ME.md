# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Information Based Inference in Models with Set-Valued Predictions and Misspecification.](http://arxiv.org/abs/2401.11046) | 本文介绍了一种基于信息的推断方法，用于处理具有集合值预测和错误规范化的模型中的部分识别参数，该方法可以在模型规范正确和错误时有效地进行推断。 |
| [^2] | [Switchback Experiments under Geometric Mixing.](http://arxiv.org/abs/2209.00197) | 本文研究了几何混合条件下的切换试验性质，并发现在该设置下，标准的切换设计受到了延续偏差的影响，但是通过谨慎使用初始燃烧期可以显著改善情况，实现误差以更慢的速度衰减。 |

# 详细

[^1]: 具有集合值预测和错误规范化的模型中的基于信息的推断方法

    Information Based Inference in Models with Set-Valued Predictions and Misspecification. (arXiv:2401.11046v1 [econ.EM])

    [http://arxiv.org/abs/2401.11046](http://arxiv.org/abs/2401.11046)

    本文介绍了一种基于信息的推断方法，用于处理具有集合值预测和错误规范化的模型中的部分识别参数，该方法可以在模型规范正确和错误时有效地进行推断。

    

    本文提出了一种基于信息的推断方法，用于不完整模型中部分识别的参数，该方法在模型规范正确和错误时均有效。该方法的关键特点是：（i）基于最小化适当定义的Kullback-Leibler信息准则，考虑模型的不完整性，并提供非空伪真集；（ii）计算可行；（iii）无论模型规范正确与否，实现方法相同；（iv）利用离散和连续协变量的变异提供的所有信息；（v）依赖于Rao的评分统计量，该统计量被证明是渐近基本的。

    This paper proposes an information-based inference method for partially identified parameters in incomplete models that is valid both when the model is correctly specified and when it is misspecified. Key features of the method are: (i) it is based on minimizing a suitably defined Kullback-Leibler information criterion that accounts for incompleteness of the model and delivers a non-empty pseudo-true set; (ii) it is computationally tractable; (iii) its implementation is the same for both correctly and incorrectly specified models; (iv) it exploits all information provided by variation in discrete and continuous covariates; (v) it relies on Rao's score statistic, which is shown to be asymptotically pivotal.
    
[^2]: 几何混合条件下的切换试验

    Switchback Experiments under Geometric Mixing. (arXiv:2209.00197v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2209.00197](http://arxiv.org/abs/2209.00197)

    本文研究了几何混合条件下的切换试验性质，并发现在该设置下，标准的切换设计受到了延续偏差的影响，但是通过谨慎使用初始燃烧期可以显著改善情况，实现误差以更慢的速度衰减。

    

    切换试验是一种通过反复对整个系统开启和关闭干预来测量治疗效果的实验设计。切换试验是克服单元间溢出效应的一种强大方法；然而，它们容易受到时间延续的偏差影响。本文研究在几何混合条件下的马尔可夫系统中的切换试验性质。我们发现，在这种情况下，标准的切换设计在延续偏差方面受到了较大的影响：它们的估计误差随着实验时间跨度$T$的增加而衰减为$T^{-1/3}$，而在没有延续效应的情况下，更快的$T^{-1/2}$衰减速度是可能的。然而，我们还展示了谨慎使用初始燃烧期可以大大改善情况，并且实现误差以$\log(T)^{1/2}T^{-1/2}$的速度衰减。我们的形式结果在实证评估中得到了验证。

    The switchback is an experimental design that measures treatment effects by repeatedly turning an intervention on and off for a whole system. Switchback experiments are a robust way to overcome cross-unit spillover effects; however, they are vulnerable to bias from temporal carryovers. In this paper, we consider properties of switchback experiments in Markovian systems that mix at a geometric rate. We find that, in this setting, standard switchback designs suffer considerably from carryover bias: Their estimation error decays as $T^{-1/3}$ in terms of the experiment horizon $T$, whereas in the absence of carryovers a faster rate of $T^{-1/2}$ would have been possible. We also show, however, that judicious use of burn-in periods can considerably improve the situation, and enables errors that decay as $\log(T)^{1/2}T^{-1/2}$. Our formal results are mirrored in an empirical evaluation.
    

