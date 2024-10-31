# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nature-Guided Cognitive Evolution for Predicting Dissolved Oxygen Concentrations in North Temperate Lakes](https://arxiv.org/abs/2403.18923) | 提出了一种自然引导的认知进化策略，通过多层融合自适应学习和自然过程，有效预测北温带湖泊中的溶解氧浓度 |
| [^2] | [Cluster-Based Normalization Layer for Neural Networks](https://arxiv.org/abs/2403.16798) | 该论文提出了一种基于聚类的神经网络规范化方法CB-Norm，通过引入高斯混合模型，解决了梯度稳定性和学习加速方面的挑战。 |
| [^3] | [Genetic-guided GFlowNets: Advancing in Practical Molecular Optimization Benchmark](https://arxiv.org/abs/2402.05961) | 本文提出了一种名为基因引导GFlowNet (Genetic GFN) 的新方法，通过集成迭代遗传搜索和训练策略，该方法在实际分子优化基准测试中取得了16.213的最新得分，明显优于现有最佳得分15.185，同时在14个任务中超越了所有对比方法。 |
| [^4] | [Set-based Neural Network Encoding.](http://arxiv.org/abs/2305.16625) | 提出了一种能够集合化地编码神经网络参数的神经网络权重编码方法，并引入了一种逐层编码方案来考虑神经网络的分层计算结构。同时引入了“pad-chunk-encode”流水线进行神经网络层的高效编码处理，还提出了新的神经网络泛化性能预测任务。 |

# 详细

[^1]: 自然引导的认知进化用于预测北温带湖泊中的溶解氧浓度

    Nature-Guided Cognitive Evolution for Predicting Dissolved Oxygen Concentrations in North Temperate Lakes

    [https://arxiv.org/abs/2403.18923](https://arxiv.org/abs/2403.18923)

    提出了一种自然引导的认知进化策略，通过多层融合自适应学习和自然过程，有效预测北温带湖泊中的溶解氧浓度

    

    预测北温带湖泊中的溶解氧（DO）浓度需要对不同生态系统中的物候模式进行全面研究，这凸显了选择物候特征和特征交互的重要性。基于过程的模型受部分过程知识限制或特征表示过于简化，而机器学习模型在有效选择不同湖泊类型和任务的相关特征交互方面面临挑战，尤其是在DO数据收集不频繁的情况下。在本文中，我们提出了一种自然引导的认知进化（NGCE）策略，这代表了自适应学习与自然过程多层融合。具体来说，我们利用代谢过程为基础的模型生成模拟DO标签。利用这些模拟标签，我们实施了一个多种群认知进化搜索，模型反映自然有机体，适应性地

    arXiv:2403.18923v1 Announce Type: cross  Abstract: Predicting dissolved oxygen (DO) concentrations in north temperate lakes requires a comprehensive study of phenological patterns across various ecosystems, which highlights the significance of selecting phenological features and feature interactions. Process-based models are limited by partial process knowledge or oversimplified feature representations, while machine learning models face challenges in efficiently selecting relevant feature interactions for different lake types and tasks, especially under the infrequent nature of DO data collection. In this paper, we propose a Nature-Guided Cognitive Evolution (NGCE) strategy, which represents a multi-level fusion of adaptive learning with natural processes. Specifically, we utilize metabolic process-based models to generate simulated DO labels. Using these simulated labels, we implement a multi-population cognitive evolutionary search, where models, mirroring natural organisms, adaptiv
    
[^2]: 基于聚类的神经网络规范化层

    Cluster-Based Normalization Layer for Neural Networks

    [https://arxiv.org/abs/2403.16798](https://arxiv.org/abs/2403.16798)

    该论文提出了一种基于聚类的神经网络规范化方法CB-Norm，通过引入高斯混合模型，解决了梯度稳定性和学习加速方面的挑战。

    

    深度学习在神经网络训练过程中面临重要挑战，包括内部协变量漂移、标签漂移、梯度消失/爆炸、过拟合和计算复杂性。传统的规范化方法，如批标准化，旨在解决其中一些问题，但通常依赖于限制其适应性的假设。混合规范化在处理多个高斯分布时面临计算障碍。本文介绍了基于聚类的规范化（CB-Norm）的两个变体——监督式基于聚类的规范化（SCB-Norm）和无监督式基于聚类的规范化（UCB-Norm），提出了一种开创性的一步规范化方法。CB-Norm利用高斯混合模型来专门解决与梯度稳定性和学习加速有关的挑战。

    arXiv:2403.16798v1 Announce Type: cross  Abstract: Deep learning faces significant challenges during the training of neural networks, including internal covariate shift, label shift, vanishing/exploding gradients, overfitting, and computational complexity. While conventional normalization methods, such as Batch Normalization, aim to tackle some of these issues, they often depend on assumptions that constrain their adaptability. Mixture Normalization faces computational hurdles in its pursuit of handling multiple Gaussian distributions.   This paper introduces Cluster-Based Normalization (CB-Norm) in two variants - Supervised Cluster-Based Normalization (SCB-Norm) and Unsupervised Cluster-Based Normalization (UCB-Norm) - proposing a groundbreaking one-step normalization approach. CB-Norm leverages a Gaussian mixture model to specifically address challenges related to gradient stability and learning acceleration.   For SCB-Norm, a supervised variant, the novel mechanism involves introduc
    
[^3]: 基因引导GFlowNets：在实际分子优化基准方面的进展

    Genetic-guided GFlowNets: Advancing in Practical Molecular Optimization Benchmark

    [https://arxiv.org/abs/2402.05961](https://arxiv.org/abs/2402.05961)

    本文提出了一种名为基因引导GFlowNet (Genetic GFN) 的新方法，通过集成迭代遗传搜索和训练策略，该方法在实际分子优化基准测试中取得了16.213的最新得分，明显优于现有最佳得分15.185，同时在14个任务中超越了所有对比方法。

    

    本文提出了一种新的GFlowNet变体，即基因引导GFlowNet (Genetic GFN)，它将迭代遗传搜索集成到GFlowNet中。遗传搜索有效地引导GFlowNet进入高回报区域，解决了全局过度探索导致的训练效率低下和探索有限区域的问题。此外，还引入了训练策略，如基于排名的重放训练和无监督最大似然预训练，以提高基因引导GFlowNet的样本效率。该方法在实际分子优化 (PMO) 领域的官方基准测试中显示了16.213的最新得分，明显优于基准测试中报告的最佳得分15.185。值得注意的是，我们的方法在23个任务中的14个任务中超过了所有对比方法，包括强化学习，贝叶斯优化，生成模型，GFlowNets和遗传算法。

    This paper proposes a novel variant of GFlowNet, genetic-guided GFlowNet (Genetic GFN), which integrates an iterative genetic search into GFlowNet. Genetic search effectively guides the GFlowNet to high-rewarded regions, addressing global over-exploration that results in training inefficiency and exploring limited regions. In addition, training strategies, such as rank-based replay training and unsupervised maximum likelihood pre-training, are further introduced to improve the sample efficiency of Genetic GFN. The proposed method shows a state-of-the-art score of 16.213, significantly outperforming the reported best score in the benchmark of 15.185, in practical molecular optimization (PMO), which is an official benchmark for sample-efficient molecular optimization. Remarkably, ours exceeds all baselines, including reinforcement learning, Bayesian optimization, generative models, GFlowNets, and genetic algorithms, in 14 out of 23 tasks.
    
[^4]: 集合化的神经网络编码

    Set-based Neural Network Encoding. (arXiv:2305.16625v1 [cs.LG])

    [http://arxiv.org/abs/2305.16625](http://arxiv.org/abs/2305.16625)

    提出了一种能够集合化地编码神经网络参数的神经网络权重编码方法，并引入了一种逐层编码方案来考虑神经网络的分层计算结构。同时引入了“pad-chunk-encode”流水线进行神经网络层的高效编码处理，还提出了新的神经网络泛化性能预测任务。

    

    我们提出了一种利用集合到集合和集合到向量函数来有效编码神经网络参数，进行泛化性能预测的神经网络权重编码方法。与之前需要对不同架构编写自定义编码模型的方法不同，我们的方法能够对混合架构和不同参数大小的模型动态编码。此外，我们的 SNE（集合化神经网络编码器）通过使用一种逐层编码方案，考虑神经网络的分层计算结构。最终将所有层次编码合并到一起，以获取神经网络编码矢量。我们还引入了“pad-chunk-encode”流水线来有效地编码神经网络层，该流水线可根据计算和内存限制进行调整。我们还引入了两个用于神经网络泛化性能预测的新任务：跨数据集和架构适应性预测。

    We propose an approach to neural network weight encoding for generalization performance prediction that utilizes set-to-set and set-to-vector functions to efficiently encode neural network parameters. Our approach is capable of encoding neural networks in a modelzoo of mixed architecture and different parameter sizes as opposed to previous approaches that require custom encoding models for different architectures. Furthermore, our \textbf{S}et-based \textbf{N}eural network \textbf{E}ncoder (SNE) takes into consideration the hierarchical computational structure of neural networks by utilizing a layer-wise encoding scheme that culminates to encoding all layer-wise encodings to obtain the neural network encoding vector. Additionally, we introduce a \textit{pad-chunk-encode} pipeline to efficiently encode neural network layers that is adjustable to computational and memory constraints. We also introduce two new tasks for neural network generalization performance prediction: cross-dataset a
    

