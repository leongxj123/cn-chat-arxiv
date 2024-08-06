# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adaptive Principal Component Regression with Applications to Panel Data.](http://arxiv.org/abs/2307.01357) | 本文提出了自适应主成分回归方法，并在面板数据中的应用中获得了均匀有限样本保证。该方法可以用于面板数据中的实验设计，特别是当干预方案是自适应分配的情况。 |
| [^2] | [Inference in Cluster Randomized Trials with Matched Pairs.](http://arxiv.org/abs/2211.14903) | 本文研究了在匹配对簇随机试验中进行统计推断的问题，提出了加权均值差估计量和方差估计量，建立了基于这些估计量的渐近精确性，并探讨了常用的两种测试程序的特性。 |

# 详细

[^1]: 自适应主成分回归在面板数据中的应用

    Adaptive Principal Component Regression with Applications to Panel Data. (arXiv:2307.01357v1 [cs.LG])

    [http://arxiv.org/abs/2307.01357](http://arxiv.org/abs/2307.01357)

    本文提出了自适应主成分回归方法，并在面板数据中的应用中获得了均匀有限样本保证。该方法可以用于面板数据中的实验设计，特别是当干预方案是自适应分配的情况。

    

    主成分回归(PCR)是一种流行的固定设计误差变量回归技术，它是线性回归的推广，观测的协变量受到随机噪声的污染。我们在数据收集时提供了在线（正则化）PCR的第一次均匀有限样本保证。由于分析固定设计中PCR的证明技术无法很容易地扩展到在线设置，我们的结果依赖于将现代鞅浓度的工具适应到误差变量设置中。作为我们界限的应用，我们在面板数据设置中提供了实验设计框架，当干预被自适应地分配时。我们的框架可以被认为是合成控制和合成干预框架的泛化，其中数据是通过自适应干预分配策略收集的。

    Principal component regression (PCR) is a popular technique for fixed-design error-in-variables regression, a generalization of the linear regression setting in which the observed covariates are corrupted with random noise. We provide the first time-uniform finite sample guarantees for online (regularized) PCR whenever data is collected adaptively. Since the proof techniques for analyzing PCR in the fixed design setting do not readily extend to the online setting, our results rely on adapting tools from modern martingale concentration to the error-in-variables setting. As an application of our bounds, we provide a framework for experiment design in panel data settings when interventions are assigned adaptively. Our framework may be thought of as a generalization of the synthetic control and synthetic interventions frameworks, where data is collected via an adaptive intervention assignment policy.
    
[^2]: 匹配对簇随机试验中的推断

    Inference in Cluster Randomized Trials with Matched Pairs. (arXiv:2211.14903v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2211.14903](http://arxiv.org/abs/2211.14903)

    本文研究了在匹配对簇随机试验中进行统计推断的问题，提出了加权均值差估计量和方差估计量，建立了基于这些估计量的渐近精确性，并探讨了常用的两种测试程序的特性。

    

    本文研究了在匹配对簇随机试验中进行推断的问题。匹配对设计意味着根据基线簇级协变量对一组簇进行匹配，然后在每对簇中随机选择一个簇进行处理。我们研究了加权均值差估计量的大样本行为，并根据匹配过程是否匹配簇大小，导出了两组不同的结果。然后，我们提出了一个方差估计量，无论匹配过程是匹配簇大小还是不匹配簇大小，都是一致的。结合这些结果，建立了基于这些估计量的检验的渐近精确性。接下来，我们考虑了基于线性回归构造的$t$测试的两种常见测试程序的特性，并声称这两种程序通常是保守的。

    This paper considers the problem of inference in cluster randomized trials where treatment status is determined according to a "matched pairs'' design. Here, by a cluster randomized experiment, we mean one in which treatment is assigned at the level of the cluster; by a "matched pairs'' design we mean that a sample of clusters is paired according to baseline, cluster-level covariates and, within each pair, one cluster is selected at random for treatment. We study the large sample behavior of a weighted difference-in-means estimator and derive two distinct sets of results depending on if the matching procedure does or does not match on cluster size. We then propose a variance estimator which is consistent in either case. Combining these results establishes the asymptotic exactness of tests based on these estimators. Next, we consider the properties of two common testing procedures based on $t$-tests constructed from linear regressions, and argue that both are generally conservative in o
    

