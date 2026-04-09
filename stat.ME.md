# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Active Statistical Inference](https://arxiv.org/abs/2403.03208) | 主动推断是一种统计推断方法，通过利用机器学习模型确定最有利于标记的数据点来有效利用预算，实现比现有基线更少样本的相同准确性。 |

# 详细

[^1]: 主动统计推断

    Active Statistical Inference

    [https://arxiv.org/abs/2403.03208](https://arxiv.org/abs/2403.03208)

    主动推断是一种统计推断方法，通过利用机器学习模型确定最有利于标记的数据点来有效利用预算，实现比现有基线更少样本的相同准确性。

    

    受主动学习概念启发，我们提出了主动推断——一种利用机器学习辅助数据收集进行统计推断的方法。假设对可收集的标签数量有预算限制，该方法利用机器学习模型确定哪些数据点最有利于标记，从而有效利用预算。其运作方式基于一种简单而强大的直觉：优先收集模型表现出不确定性的数据点的标签，并在模型表现出自信时依赖于其预测。主动推断构建了可证明有效的置信区间和假设检验，同时利用任何黑盒机器学习模型并处理任何数据分布。关键点在于，它能以比依赖于非自适应收集数据的现有基线更少的样本达到相同水平的准确性。这意味着对于相同数量的样本，...

    arXiv:2403.03208v1 Announce Type: cross  Abstract: Inspired by the concept of active learning, we propose active inference$\unicode{x2013}$a methodology for statistical inference with machine-learning-assisted data collection. Assuming a budget on the number of labels that can be collected, the methodology uses a machine learning model to identify which data points would be most beneficial to label, thus effectively utilizing the budget. It operates on a simple yet powerful intuition: prioritize the collection of labels for data points where the model exhibits uncertainty, and rely on the model's predictions where it is confident. Active inference constructs provably valid confidence intervals and hypothesis tests while leveraging any black-box machine learning model and handling any data distribution. The key point is that it achieves the same level of accuracy with far fewer samples than existing baselines relying on non-adaptively-collected data. This means that for the same number 
    

