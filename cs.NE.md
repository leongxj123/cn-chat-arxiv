# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Genetic-guided GFlowNets: Advancing in Practical Molecular Optimization Benchmark](https://arxiv.org/abs/2402.05961) | 本文提出了一种名为基因引导GFlowNet (Genetic GFN) 的新方法，通过集成迭代遗传搜索和训练策略，该方法在实际分子优化基准测试中取得了16.213的最新得分，明显优于现有最佳得分15.185，同时在14个任务中超越了所有对比方法。 |
| [^2] | [Deep Kalman Filters Can Filter.](http://arxiv.org/abs/2310.19603) | 本研究展示了一类连续时间的深度卡尔曼滤波器（DKFs），可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而具有在数学金融领域中传统模型基础上的滤波问题的应用潜力。 |

# 详细

[^1]: 基因引导GFlowNets：在实际分子优化基准方面的进展

    Genetic-guided GFlowNets: Advancing in Practical Molecular Optimization Benchmark

    [https://arxiv.org/abs/2402.05961](https://arxiv.org/abs/2402.05961)

    本文提出了一种名为基因引导GFlowNet (Genetic GFN) 的新方法，通过集成迭代遗传搜索和训练策略，该方法在实际分子优化基准测试中取得了16.213的最新得分，明显优于现有最佳得分15.185，同时在14个任务中超越了所有对比方法。

    

    本文提出了一种新的GFlowNet变体，即基因引导GFlowNet (Genetic GFN)，它将迭代遗传搜索集成到GFlowNet中。遗传搜索有效地引导GFlowNet进入高回报区域，解决了全局过度探索导致的训练效率低下和探索有限区域的问题。此外，还引入了训练策略，如基于排名的重放训练和无监督最大似然预训练，以提高基因引导GFlowNet的样本效率。该方法在实际分子优化 (PMO) 领域的官方基准测试中显示了16.213的最新得分，明显优于基准测试中报告的最佳得分15.185。值得注意的是，我们的方法在23个任务中的14个任务中超过了所有对比方法，包括强化学习，贝叶斯优化，生成模型，GFlowNets和遗传算法。

    This paper proposes a novel variant of GFlowNet, genetic-guided GFlowNet (Genetic GFN), which integrates an iterative genetic search into GFlowNet. Genetic search effectively guides the GFlowNet to high-rewarded regions, addressing global over-exploration that results in training inefficiency and exploring limited regions. In addition, training strategies, such as rank-based replay training and unsupervised maximum likelihood pre-training, are further introduced to improve the sample efficiency of Genetic GFN. The proposed method shows a state-of-the-art score of 16.213, significantly outperforming the reported best score in the benchmark of 15.185, in practical molecular optimization (PMO), which is an official benchmark for sample-efficient molecular optimization. Remarkably, ours exceeds all baselines, including reinforcement learning, Bayesian optimization, generative models, GFlowNets, and genetic algorithms, in 14 out of 23 tasks.
    
[^2]: 深度卡尔曼滤波器可以进行滤波

    Deep Kalman Filters Can Filter. (arXiv:2310.19603v1 [cs.LG])

    [http://arxiv.org/abs/2310.19603](http://arxiv.org/abs/2310.19603)

    本研究展示了一类连续时间的深度卡尔曼滤波器（DKFs），可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而具有在数学金融领域中传统模型基础上的滤波问题的应用潜力。

    

    深度卡尔曼滤波器（DKFs）是一类神经网络模型，可以从序列数据中生成高斯概率测度。虽然DKFs受卡尔曼滤波器的启发，但它们缺乏与随机滤波问题的具体理论关联，从而限制了它们在传统模型基础上的滤波问题的应用，例如数学金融中的债券和期权定价模型校准。我们通过展示一类连续时间DKFs，可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而解决了深度学习数学基础中的这个问题。我们的近似结果在路径的足够规则的紧致子集上一致成立，其中近似误差由在给定紧致路径集上均一地计算的最坏情况2-Wasserstein距离量化。

    Deep Kalman filters (DKFs) are a class of neural network models that generate Gaussian probability measures from sequential data. Though DKFs are inspired by the Kalman filter, they lack concrete theoretical ties to the stochastic filtering problem, thus limiting their applicability to areas where traditional model-based filters have been used, e.g.\ model calibration for bond and option prices in mathematical finance. We address this issue in the mathematical foundations of deep learning by exhibiting a class of continuous-time DKFs which can approximately implement the conditional law of a broad class of non-Markovian and conditionally Gaussian signal processes given noisy continuous-times measurements. Our approximation results hold uniformly over sufficiently regular compact subsets of paths, where the approximation error is quantified by the worst-case 2-Wasserstein distance computed uniformly over the given compact set of paths.
    

