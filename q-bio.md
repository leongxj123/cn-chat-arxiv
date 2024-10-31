# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Genetic-guided GFlowNets: Advancing in Practical Molecular Optimization Benchmark](https://arxiv.org/abs/2402.05961) | 本文提出了一种名为基因引导GFlowNet (Genetic GFN) 的新方法，通过集成迭代遗传搜索和训练策略，该方法在实际分子优化基准测试中取得了16.213的最新得分，明显优于现有最佳得分15.185，同时在14个任务中超越了所有对比方法。 |
| [^2] | [Unlocking the Power of Multi-institutional Data: Integrating and Harmonizing Genomic Data Across Institutions](https://arxiv.org/abs/2402.00077) | 该研究介绍了一种名为Bridge的模型，致力于解决利用多机构测序数据时面临的挑战，包括基因组板块的变化、测序技术的差异以及数据的高维度和稀疏性等。 |

# 详细

[^1]: 基因引导GFlowNets：在实际分子优化基准方面的进展

    Genetic-guided GFlowNets: Advancing in Practical Molecular Optimization Benchmark

    [https://arxiv.org/abs/2402.05961](https://arxiv.org/abs/2402.05961)

    本文提出了一种名为基因引导GFlowNet (Genetic GFN) 的新方法，通过集成迭代遗传搜索和训练策略，该方法在实际分子优化基准测试中取得了16.213的最新得分，明显优于现有最佳得分15.185，同时在14个任务中超越了所有对比方法。

    

    本文提出了一种新的GFlowNet变体，即基因引导GFlowNet (Genetic GFN)，它将迭代遗传搜索集成到GFlowNet中。遗传搜索有效地引导GFlowNet进入高回报区域，解决了全局过度探索导致的训练效率低下和探索有限区域的问题。此外，还引入了训练策略，如基于排名的重放训练和无监督最大似然预训练，以提高基因引导GFlowNet的样本效率。该方法在实际分子优化 (PMO) 领域的官方基准测试中显示了16.213的最新得分，明显优于基准测试中报告的最佳得分15.185。值得注意的是，我们的方法在23个任务中的14个任务中超过了所有对比方法，包括强化学习，贝叶斯优化，生成模型，GFlowNets和遗传算法。

    This paper proposes a novel variant of GFlowNet, genetic-guided GFlowNet (Genetic GFN), which integrates an iterative genetic search into GFlowNet. Genetic search effectively guides the GFlowNet to high-rewarded regions, addressing global over-exploration that results in training inefficiency and exploring limited regions. In addition, training strategies, such as rank-based replay training and unsupervised maximum likelihood pre-training, are further introduced to improve the sample efficiency of Genetic GFN. The proposed method shows a state-of-the-art score of 16.213, significantly outperforming the reported best score in the benchmark of 15.185, in practical molecular optimization (PMO), which is an official benchmark for sample-efficient molecular optimization. Remarkably, ours exceeds all baselines, including reinforcement learning, Bayesian optimization, generative models, GFlowNets, and genetic algorithms, in 14 out of 23 tasks.
    
[^2]: 多机构数据的释放力量：整合和协调跨机构的基因组数据

    Unlocking the Power of Multi-institutional Data: Integrating and Harmonizing Genomic Data Across Institutions

    [https://arxiv.org/abs/2402.00077](https://arxiv.org/abs/2402.00077)

    该研究介绍了一种名为Bridge的模型，致力于解决利用多机构测序数据时面临的挑战，包括基因组板块的变化、测序技术的差异以及数据的高维度和稀疏性等。

    

    癌症是由基因突变驱动的复杂疾病，肿瘤测序已成为癌症患者临床护理的重要手段。出现的多机构测序数据为学习真实世界的证据以增强精准肿瘤医学提供了强大的资源。由美国癌症研究协会领导的GENIE BPC建立了一个独特的数据库，将基因组数据与多个癌症中心的临床信息联系起来。然而，利用这种多机构测序数据面临着重大挑战。基因组板块的变化导致在使用常见基因集进行分析时信息丢失。此外，不同的测序技术和机构之间的患者异质性增加了复杂性。高维数据、稀疏基因突变模式以及个体基因水平上的弱信号进一步增加了问题的复杂性。在这些现实世界的挑战的推动下，我们引入了Bridge模型。

    Cancer is a complex disease driven by genomic alterations, and tumor sequencing is becoming a mainstay of clinical care for cancer patients. The emergence of multi-institution sequencing data presents a powerful resource for learning real-world evidence to enhance precision oncology. GENIE BPC, led by the American Association for Cancer Research, establishes a unique database linking genomic data with clinical information for patients treated at multiple cancer centers. However, leveraging such multi-institutional sequencing data presents significant challenges. Variations in gene panels result in loss of information when the analysis is conducted on common gene sets. Additionally, differences in sequencing techniques and patient heterogeneity across institutions add complexity. High data dimensionality, sparse gene mutation patterns, and weak signals at the individual gene level further complicate matters. Motivated by these real-world challenges, we introduce the Bridge model. It use
    

