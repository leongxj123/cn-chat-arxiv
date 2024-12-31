# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Genetic-guided GFlowNets: Advancing in Practical Molecular Optimization Benchmark](https://arxiv.org/abs/2402.05961) | 本文提出了一种名为基因引导GFlowNet (Genetic GFN) 的新方法，通过集成迭代遗传搜索和训练策略，该方法在实际分子优化基准测试中取得了16.213的最新得分，明显优于现有最佳得分15.185，同时在14个任务中超越了所有对比方法。 |

# 详细

[^1]: 基因引导GFlowNets：在实际分子优化基准方面的进展

    Genetic-guided GFlowNets: Advancing in Practical Molecular Optimization Benchmark

    [https://arxiv.org/abs/2402.05961](https://arxiv.org/abs/2402.05961)

    本文提出了一种名为基因引导GFlowNet (Genetic GFN) 的新方法，通过集成迭代遗传搜索和训练策略，该方法在实际分子优化基准测试中取得了16.213的最新得分，明显优于现有最佳得分15.185，同时在14个任务中超越了所有对比方法。

    

    本文提出了一种新的GFlowNet变体，即基因引导GFlowNet (Genetic GFN)，它将迭代遗传搜索集成到GFlowNet中。遗传搜索有效地引导GFlowNet进入高回报区域，解决了全局过度探索导致的训练效率低下和探索有限区域的问题。此外，还引入了训练策略，如基于排名的重放训练和无监督最大似然预训练，以提高基因引导GFlowNet的样本效率。该方法在实际分子优化 (PMO) 领域的官方基准测试中显示了16.213的最新得分，明显优于基准测试中报告的最佳得分15.185。值得注意的是，我们的方法在23个任务中的14个任务中超过了所有对比方法，包括强化学习，贝叶斯优化，生成模型，GFlowNets和遗传算法。

    This paper proposes a novel variant of GFlowNet, genetic-guided GFlowNet (Genetic GFN), which integrates an iterative genetic search into GFlowNet. Genetic search effectively guides the GFlowNet to high-rewarded regions, addressing global over-exploration that results in training inefficiency and exploring limited regions. In addition, training strategies, such as rank-based replay training and unsupervised maximum likelihood pre-training, are further introduced to improve the sample efficiency of Genetic GFN. The proposed method shows a state-of-the-art score of 16.213, significantly outperforming the reported best score in the benchmark of 15.185, in practical molecular optimization (PMO), which is an official benchmark for sample-efficient molecular optimization. Remarkably, ours exceeds all baselines, including reinforcement learning, Bayesian optimization, generative models, GFlowNets, and genetic algorithms, in 14 out of 23 tasks.
    

