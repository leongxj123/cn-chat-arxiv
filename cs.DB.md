# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Sampling-based Framework for Hypothesis Testing on Large Attributed Graphs](https://arxiv.org/abs/2403.13286) | 本论文提出了一个基于抽样的假设检验框架，能够在大属性图中处理节点、边和路径假设，通过提出路径假设感知采样器 PHASE 以及 PHASEopt，实现了准确且高效的抽样，实验证明了其在假设检验上的优势。 |

# 详细

[^1]: 基于抽样的大属性图假设检验框架

    A Sampling-based Framework for Hypothesis Testing on Large Attributed Graphs

    [https://arxiv.org/abs/2403.13286](https://arxiv.org/abs/2403.13286)

    本论文提出了一个基于抽样的假设检验框架，能够在大属性图中处理节点、边和路径假设，通过提出路径假设感知采样器 PHASE 以及 PHASEopt，实现了准确且高效的抽样，实验证明了其在假设检验上的优势。

    

    假设检验是一种用于从样本数据中得出关于总体的结论的统计方法，通常用表格表示。随着现实应用中图表示的普及，图中的假设检验变得越来越重要。本文对属性图中的节点、边和路径假设进行了形式化。我们开发了一个基于抽样的假设检验框架，可以容纳现有的假设不可知的图抽样方法。为了实现准确和高效的抽样，我们提出了一种路径假设感知采样器 PHASE，它是一种考虑假设中指定路径的 m-维随机游走。我们进一步优化了其时间效率并提出了 PHASEopt。对真实数据集的实验表明，我们的框架能够利用常见的图抽样方法进行假设检验，并且在准确性和时间效率方面假设感知抽样具有优势。

    arXiv:2403.13286v1 Announce Type: cross  Abstract: Hypothesis testing is a statistical method used to draw conclusions about populations from sample data, typically represented in tables. With the prevalence of graph representations in real-life applications, hypothesis testing in graphs is gaining importance. In this work, we formalize node, edge, and path hypotheses in attributed graphs. We develop a sampling-based hypothesis testing framework, which can accommodate existing hypothesis-agnostic graph sampling methods. To achieve accurate and efficient sampling, we then propose a Path-Hypothesis-Aware SamplEr, PHASE, an m- dimensional random walk that accounts for the paths specified in a hypothesis. We further optimize its time efficiency and propose PHASEopt. Experiments on real datasets demonstrate the ability of our framework to leverage common graph sampling methods for hypothesis testing, and the superiority of hypothesis-aware sampling in terms of accuracy and time efficiency.
    

