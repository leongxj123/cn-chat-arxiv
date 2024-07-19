# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scalable Spatiotemporal Prediction with Bayesian Neural Fields](https://arxiv.org/abs/2403.07657) | 该论文提出了贝叶斯神经场（BayesNF），结合了深度神经网络和分层贝叶斯推断，用于处理大规模时空预测问题。 |
| [^2] | [Bayesian sequential design of computer experiments for quantile set inversion.](http://arxiv.org/abs/2211.01008) | 本论文提出了一种基于贝叶斯策略的量化集反演方法，通过高斯过程建模和逐步不确定性减少原理，顺序选择评估函数的点，从而有效近似感兴趣的集合。 |

# 详细

[^1]: 使用贝叶斯神经场进行可扩展的时空预测

    Scalable Spatiotemporal Prediction with Bayesian Neural Fields

    [https://arxiv.org/abs/2403.07657](https://arxiv.org/abs/2403.07657)

    该论文提出了贝叶斯神经场（BayesNF），结合了深度神经网络和分层贝叶斯推断，用于处理大规模时空预测问题。

    

    时空数据集由空间参考的时间序列表示，广泛应用于许多科学和商业智能领域，例如空气污染监测，疾病跟踪和云需求预测。随着现代数据集规模和复杂性的不断增加，需要新的统计方法来捕捉复杂的时空动态并处理大规模预测问题。本研究介绍了Bayesian Neural Field (BayesNF)，这是一个用于推断时空域上丰富概率分布的通用领域统计模型，可用于包括预测、插值和变异分析在内的数据分析任务。BayesNF将用于高容量函数估计的新型深度神经网络架构与用于鲁棒不确定性量化的分层贝叶斯推断相结合。通过在定义先验分布方面进行序列化

    arXiv:2403.07657v1 Announce Type: cross  Abstract: Spatiotemporal datasets, which consist of spatially-referenced time series, are ubiquitous in many scientific and business-intelligence applications, such as air pollution monitoring, disease tracking, and cloud-demand forecasting. As modern datasets continue to increase in size and complexity, there is a growing need for new statistical methods that are flexible enough to capture complex spatiotemporal dynamics and scalable enough to handle large prediction problems. This work presents the Bayesian Neural Field (BayesNF), a domain-general statistical model for inferring rich probability distributions over a spatiotemporal domain, which can be used for data-analysis tasks including forecasting, interpolation, and variography. BayesNF integrates a novel deep neural network architecture for high-capacity function estimation with hierarchical Bayesian inference for robust uncertainty quantification. By defining the prior through a sequenc
    
[^2]: 基于贝叶斯序贯设计的计算机实验量化集反演

    Bayesian sequential design of computer experiments for quantile set inversion. (arXiv:2211.01008v2 [stat.ML] CROSS LISTED)

    [http://arxiv.org/abs/2211.01008](http://arxiv.org/abs/2211.01008)

    本论文提出了一种基于贝叶斯策略的量化集反演方法，通过高斯过程建模和逐步不确定性减少原理，顺序选择评估函数的点，从而有效近似感兴趣的集合。

    

    我们考虑一个未知的多元函数，它代表着一个系统，如一个复杂的数值模拟器，同时具有确定性和不确定性的输入。我们的目标是估计确定性输入集，这些输入导致的输出（就不确定性输入的分布而言）属于给定集合的概率小于给定阈值。这个问题被称为量化集反演（QSI），例如在稳健（基于可靠性）优化问题的背景下，当寻找满足约束条件且具有足够大概率的解集时会发生。为了解决QSI问题，我们提出了一种基于高斯过程建模和逐步不确定性减少（SUR）原理的贝叶斯策略，以顺序选择应该评估函数的点，以便高效近似感兴趣的集合。通过几个数值实验，我们展示了所提出的SUR策略的性能和价值

    We consider an unknown multivariate function representing a system-such as a complex numerical simulator-taking both deterministic and uncertain inputs. Our objective is to estimate the set of deterministic inputs leading to outputs whose probability (with respect to the distribution of the uncertain inputs) of belonging to a given set is less than a given threshold. This problem, which we call Quantile Set Inversion (QSI), occurs for instance in the context of robust (reliability-based) optimization problems, when looking for the set of solutions that satisfy the constraints with sufficiently large probability. To solve the QSI problem, we propose a Bayesian strategy based on Gaussian process modeling and the Stepwise Uncertainty Reduction (SUR) principle, to sequentially choose the points at which the function should be evaluated to efficiently approximate the set of interest. We illustrate the performance and interest of the proposed SUR strategy through several numerical experiment
    

