# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Automatic Outlier Rectification via Optimal Transport](https://arxiv.org/abs/2403.14067) | 提出了一种自动异常值矫正机制，通过将矫正和估计集成到联合优化框架中，利用最优输运和凹成本函数来检测和移除异常值，并选择最佳分布来执行估计任务 |
| [^2] | [Adjustment Identification Distance: A gadjid for Causal Structure Learning](https://arxiv.org/abs/2402.08616) | gadjid软件包提供了一种用于因果结构学习的调整识别距离，通过引入框架来计算因果距离，这些距离能够高效评估因果发现算法学习的图形，并且在处理大规模图形时具有较高的性能。 |

# 详细

[^1]: 通过最优输运的自动异常值矫正

    Automatic Outlier Rectification via Optimal Transport

    [https://arxiv.org/abs/2403.14067](https://arxiv.org/abs/2403.14067)

    提出了一种自动异常值矫正机制，通过将矫正和估计集成到联合优化框架中，利用最优输运和凹成本函数来检测和移除异常值，并选择最佳分布来执行估计任务

    

    在本文中，我们提出了一个新颖的概念框架，使用具有凹成本函数的最优输运来检测异常值。传统的异常值检测方法通常使用两阶段流程：首先检测并移除异常值，然后在清洁数据上执行估计。然而，这种方法并没有将异常值移除与估计任务联系起来，留下了改进的空间。为了解决这一局限性，我们提出了一种自动异常值矫正机制，将矫正和估计集成到一个联合优化框架中。我们首先利用具有凹成本函数的最优输运距离来构建概率分布空间中的矫正集合。然后，我们选择在矫正集合中的最佳分布来执行估计任务。值得注意的是，我们在本文中引入的凹成本函数是使我们的估计器具有关键性的因素。

    arXiv:2403.14067v1 Announce Type: cross  Abstract: In this paper, we propose a novel conceptual framework to detect outliers using optimal transport with a concave cost function. Conventional outlier detection approaches typically use a two-stage procedure: first, outliers are detected and removed, and then estimation is performed on the cleaned data. However, this approach does not inform outlier removal with the estimation task, leaving room for improvement. To address this limitation, we propose an automatic outlier rectification mechanism that integrates rectification and estimation within a joint optimization framework. We take the first step to utilize an optimal transport distance with a concave cost function to construct a rectification set in the space of probability distributions. Then, we select the best distribution within the rectification set to perform the estimation task. Notably, the concave cost function we introduced in this paper is the key to making our estimator e
    
[^2]: Adjustment Identification Distance: 一种用于因果结构学习的调整识别距离

    Adjustment Identification Distance: A gadjid for Causal Structure Learning

    [https://arxiv.org/abs/2402.08616](https://arxiv.org/abs/2402.08616)

    gadjid软件包提供了一种用于因果结构学习的调整识别距离，通过引入框架来计算因果距离，这些距离能够高效评估因果发现算法学习的图形，并且在处理大规模图形时具有较高的性能。

    

    通过因果发现算法学习的图形的评估是困难的：两个图形之间不同的边的数量不能反映出它们在建议因果效应的识别公式方面有何不同。我们引入了一个框架，用于开发图形之间的因果距离，其中包括有向无环图的结构干预距离作为一种特殊情况。我们利用这个框架开发了改进的基于调整的距离，以及对完成的部分有向无环图和因果序列的扩展。我们开发了多项式时间可达性算法来高效计算距离。在我们的gadjid软件包中（在https://github.com/CausalDisco/gadjid上开源），我们提供了我们的距离实现；它们的运行速度比结构干预距离快几个数量级，从而为以前无法扩展的图形尺寸提供了一个因果发现的成功指标。

    Evaluating graphs learned by causal discovery algorithms is difficult: The number of edges that differ between two graphs does not reflect how the graphs differ with respect to the identifying formulas they suggest for causal effects. We introduce a framework for developing causal distances between graphs which includes the structural intervention distance for directed acyclic graphs as a special case. We use this framework to develop improved adjustment-based distances as well as extensions to completed partially directed acyclic graphs and causal orders. We develop polynomial-time reachability algorithms to compute the distances efficiently. In our package gadjid (open source at https://github.com/CausalDisco/gadjid), we provide implementations of our distances; they are orders of magnitude faster than the structural intervention distance and thereby provide a success metric for causal discovery that scales to graph sizes that were previously prohibitive.
    

