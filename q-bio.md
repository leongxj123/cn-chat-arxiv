# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Genetic-guided GFlowNets: Advancing in Practical Molecular Optimization Benchmark](https://arxiv.org/abs/2402.05961) | 本文提出了一种名为基因引导GFlowNet (Genetic GFN) 的新方法，通过集成迭代遗传搜索和训练策略，该方法在实际分子优化基准测试中取得了16.213的最新得分，明显优于现有最佳得分15.185，同时在14个任务中超越了所有对比方法。 |
| [^2] | [Simulation-based Inference for Cardiovascular Models.](http://arxiv.org/abs/2307.13918) | 本研究将心血管模型的逆问题作为统计推理进行解决，在体外进行了五个生物标记物的不确定性分析，展示了模拟推理的能力。 |
| [^3] | [Deep Neural Networks and Brain Alignment: Brain Encoding and Decoding (Survey).](http://arxiv.org/abs/2307.10246) | 本文综述了深度神经网络和脑对齐的研究，重点在于脑编码和解码模型的应用。这些模型对于理解大脑的信息处理机制以及设计脑机接口具有重要意义。 |

# 详细

[^1]: 基因引导GFlowNets：在实际分子优化基准方面的进展

    Genetic-guided GFlowNets: Advancing in Practical Molecular Optimization Benchmark

    [https://arxiv.org/abs/2402.05961](https://arxiv.org/abs/2402.05961)

    本文提出了一种名为基因引导GFlowNet (Genetic GFN) 的新方法，通过集成迭代遗传搜索和训练策略，该方法在实际分子优化基准测试中取得了16.213的最新得分，明显优于现有最佳得分15.185，同时在14个任务中超越了所有对比方法。

    

    本文提出了一种新的GFlowNet变体，即基因引导GFlowNet (Genetic GFN)，它将迭代遗传搜索集成到GFlowNet中。遗传搜索有效地引导GFlowNet进入高回报区域，解决了全局过度探索导致的训练效率低下和探索有限区域的问题。此外，还引入了训练策略，如基于排名的重放训练和无监督最大似然预训练，以提高基因引导GFlowNet的样本效率。该方法在实际分子优化 (PMO) 领域的官方基准测试中显示了16.213的最新得分，明显优于基准测试中报告的最佳得分15.185。值得注意的是，我们的方法在23个任务中的14个任务中超过了所有对比方法，包括强化学习，贝叶斯优化，生成模型，GFlowNets和遗传算法。

    This paper proposes a novel variant of GFlowNet, genetic-guided GFlowNet (Genetic GFN), which integrates an iterative genetic search into GFlowNet. Genetic search effectively guides the GFlowNet to high-rewarded regions, addressing global over-exploration that results in training inefficiency and exploring limited regions. In addition, training strategies, such as rank-based replay training and unsupervised maximum likelihood pre-training, are further introduced to improve the sample efficiency of Genetic GFN. The proposed method shows a state-of-the-art score of 16.213, significantly outperforming the reported best score in the benchmark of 15.185, in practical molecular optimization (PMO), which is an official benchmark for sample-efficient molecular optimization. Remarkably, ours exceeds all baselines, including reinforcement learning, Bayesian optimization, generative models, GFlowNets, and genetic algorithms, in 14 out of 23 tasks.
    
[^2]: 基于模拟的推理用于心血管模型

    Simulation-based Inference for Cardiovascular Models. (arXiv:2307.13918v1 [stat.ML])

    [http://arxiv.org/abs/2307.13918](http://arxiv.org/abs/2307.13918)

    本研究将心血管模型的逆问题作为统计推理进行解决，在体外进行了五个生物标记物的不确定性分析，展示了模拟推理的能力。

    

    在过去的几十年中，血流动力学模拟器不断发展，已成为研究体外心血管系统的首选工具。虽然这样的工具通常用于从生理参数模拟全身血流动力学，但解决将波形映射回合理的生理参数的逆问题仍然有很大的潜力和挑战。受模拟推理（SBI）的进展的启发，我们将这个逆问题作为统计推理来处理。与其他方法不同，SBI为感兴趣的参数提供了后验分布，提供了关于个体测量的不确定性的多维表示。我们通过对比几种测量模态来展示这种能力，进行了五个临床感兴趣的生物标志物的体外不确定性分析。除了确认已知事实，比如估计心率的可行性，我们的研究还突出了…

    Over the past decades, hemodynamics simulators have steadily evolved and have become tools of choice for studying cardiovascular systems in-silico. While such tools are routinely used to simulate whole-body hemodynamics from physiological parameters, solving the corresponding inverse problem of mapping waveforms back to plausible physiological parameters remains both promising and challenging. Motivated by advances in simulation-based inference (SBI), we cast this inverse problem as statistical inference. In contrast to alternative approaches, SBI provides \textit{posterior distributions} for the parameters of interest, providing a \textit{multi-dimensional} representation of uncertainty for \textit{individual} measurements. We showcase this ability by performing an in-silico uncertainty analysis of five biomarkers of clinical interest comparing several measurement modalities. Beyond the corroboration of known facts, such as the feasibility of estimating heart rate, our study highlight
    
[^3]: 深度神经网络和脑对齐：脑编码和解码（综述）

    Deep Neural Networks and Brain Alignment: Brain Encoding and Decoding (Survey). (arXiv:2307.10246v1 [q-bio.NC])

    [http://arxiv.org/abs/2307.10246](http://arxiv.org/abs/2307.10246)

    本文综述了深度神经网络和脑对齐的研究，重点在于脑编码和解码模型的应用。这些模型对于理解大脑的信息处理机制以及设计脑机接口具有重要意义。

    

    大脑如何表示不同的信息模式？我们能否设计出一个可以自动理解用户思考内容的系统？这些问题可以通过研究功能磁共振成像（fMRI）等大脑记录来回答。作为第一步，神经科学界为被动阅读/听觉/观看概念词汇、叙述、图片和电影相关的认知神经科学数据集作出了贡献。过去二十年中，还提出了使用这些数据集的编码和解码模型。这些模型作为基础研究中的额外工具，在认知科学和神经科学领域有着多种实际应用。编码模型旨在自动地生成fMRI大脑表征，给定一个刺激。它们在评估和诊断神经系统疾病以及设计大脑损伤治疗方法方面有着多种实际应用。解码模型解决了根据fMRI重构刺激的逆问题。它们对于理解大脑如何处理信息以及设计脑机接口的发展都有着重要意义。

    How does the brain represent different modes of information? Can we design a system that automatically understands what the user is thinking? Such questions can be answered by studying brain recordings like functional magnetic resonance imaging (fMRI). As a first step, the neuroscience community has contributed several large cognitive neuroscience datasets related to passive reading/listening/viewing of concept words, narratives, pictures and movies. Encoding and decoding models using these datasets have also been proposed in the past two decades. These models serve as additional tools for basic research in cognitive science and neuroscience. Encoding models aim at generating fMRI brain representations given a stimulus automatically. They have several practical applications in evaluating and diagnosing neurological conditions and thus also help design therapies for brain damage. Decoding models solve the inverse problem of reconstructing the stimuli given the fMRI. They are useful for 
    

