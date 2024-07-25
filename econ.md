# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Binscatter Regressions.](http://arxiv.org/abs/1902.09615) | 本论文介绍了Stata包Binsreg，该包实现了Binscatter方法，并提供了七个命令来进行回归分析。这些命令能够进行点估计、不确定性量化和推断测试，同时提供了分组散点图和多组统计比较。 |

# 详细

[^1]: Binscatter回归

    Binscatter Regressions. (arXiv:1902.09615v4 [econ.EM] UPDATED)

    [http://arxiv.org/abs/1902.09615](http://arxiv.org/abs/1902.09615)

    本论文介绍了Stata包Binsreg，该包实现了Binscatter方法，并提供了七个命令来进行回归分析。这些命令能够进行点估计、不确定性量化和推断测试，同时提供了分组散点图和多组统计比较。

    

    我们介绍了Stata软件包Binsreg，该软件包实现了Cattaneo、Crump、Farrell和Feng（2023a,b）开发的binscatter方法。该软件包包括七个命令：binsreg、binslogit、binsprobit、binsqreg、binstest、binspwc和binsregselect。前四个命令实现了基本和扩展的最小二乘binscatter回归（binsreg）的点估计和不确定性量化（置信区间和置信带），以及广义非线性binscatter回归（binslogit用于Logit回归，binsprobit用于Probit回归，binsqreg用于分位数回归）。这些命令还提供了分组散点图，适用于单样本和多样本设置。后两个命令集中在点态和一致性推断方面：binstest实施了关于参数设定和未知回归函数的非参数形状限制的假设检验程序，而binspwc实施了多组成对统计比较。

    We introduce the Stata package Binsreg, which implements the binscatter methods developed in Cattaneo, Crump, Farrell and Feng (2023a,b). The package includes seven commands: binsreg, binslogit, binsprobit, binsqreg, binstest, binspwc, and binsregselect. The first four commands implement point estimation and uncertainty quantification (confidence intervals and confidence bands) for canonical and extended least squares binscatter regression (binsreg) as well as generalized nonlinear binscatter regression (binslogit for Logit regression, binsprobit for Probit regression, and binsqreg for quantile regression). These commands also offer binned scatter plots, allowing for one- and multi-sample settings. The next two commands focus on pointwise and uniform inference: binstest implements hypothesis testing procedures for parametric specifications and for nonparametric shape restrictions of the unknown regression function, while binspwc implements multi-group pairwise statistical comparisons. 
    

