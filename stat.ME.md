# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sequential Kernelized Independence Testing.](http://arxiv.org/abs/2212.07383) | 该论文介绍了顺序核独立性测试的方法，以解决传统批量测试在流数据上的问题，实现了根据任务复杂性自适应调整样本大小，并在收集新数据后持续监测和控制误报率。 |

# 详细

[^1]: 顺序核独立性测试

    Sequential Kernelized Independence Testing. (arXiv:2212.07383v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2212.07383](http://arxiv.org/abs/2212.07383)

    该论文介绍了顺序核独立性测试的方法，以解决传统批量测试在流数据上的问题，实现了根据任务复杂性自适应调整样本大小，并在收集新数据后持续监测和控制误报率。

    

    独立性测试是一个经典的统计问题，在固定采集数据之前的批量设置中得到了广泛研究。然而，实践者们往往更喜欢能够根据问题的复杂性进行自适应的程序，而不是事先设定样本大小。理想情况下，这样的程序应该（a）在简单任务上尽早停止（在困难任务上稍后停止），因此更好地利用可用资源，以及（b）在收集新数据之后，持续监测数据并高效地整合统计证据，同时控制误报率。经典的批量测试不适用于流数据：在数据观察后进行有效推断需要对多重测试进行校正，这导致了低功率。遵循通过投注进行测试的原则，我们设计了顺序核独立性测试，克服了这些缺点。我们通过采用由核相关性测度（如Hilbert-）启发的投注来说明我们的广泛框架。

    Independence testing is a classical statistical problem that has been extensively studied in the batch setting when one fixes the sample size before collecting data. However, practitioners often prefer procedures that adapt to the complexity of a problem at hand instead of setting sample size in advance. Ideally, such procedures should (a) stop earlier on easy tasks (and later on harder tasks), hence making better use of available resources, and (b) continuously monitor the data and efficiently incorporate statistical evidence after collecting new data, while controlling the false alarm rate. Classical batch tests are not tailored for streaming data: valid inference after data peeking requires correcting for multiple testing which results in low power. Following the principle of testing by betting, we design sequential kernelized independence tests that overcome such shortcomings. We exemplify our broad framework using bets inspired by kernelized dependence measures, e.g., the Hilbert-
    

