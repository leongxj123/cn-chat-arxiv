# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Uncertainty, Calibration, and Membership Inference Attacks: An Information-Theoretic Perspective](https://arxiv.org/abs/2402.10686) | 通过信息论框架分析了最先进的似然比攻击对不确定性、校准水平和数据集大小的影响，研究了成员推理攻击中隐含的风险 |
| [^2] | [Disentanglement Learning via Topology.](http://arxiv.org/abs/2308.12696) | 本文提出了一种通过拓扑损失实现解缠编码的方法，这是第一个提出用于解缠的可微拓扑损失的论文，实验结果表明所提出的方法相对于最新结果改进了解缠得分。 |
| [^3] | [Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts.](http://arxiv.org/abs/2306.04723) | 本文提出了衡量文本内部维度的方法，应用于鲁棒性AI生成文本的检测，展示了人类文本与AI生成文本在内部维度上的差异。 |

# 详细

[^1]: 不确定性、校准和成员推理攻击：信息论视角

    Uncertainty, Calibration, and Membership Inference Attacks: An Information-Theoretic Perspective

    [https://arxiv.org/abs/2402.10686](https://arxiv.org/abs/2402.10686)

    通过信息论框架分析了最先进的似然比攻击对不确定性、校准水平和数据集大小的影响，研究了成员推理攻击中隐含的风险

    

    在成员推理攻击（MIA）中，攻击者利用典型机器学习模型表现出的过度自信来确定特定数据点是否被用于训练目标模型。在本文中，我们在一个信息理论框架内分析了最先进的似然比攻击（LiRA）的性能，这个框架可以允许研究真实数据生成过程中的不确定性的影响，由有限训练数据集引起的认知不确定性以及目标模型的校准水平。我们比较了三种不同的设置，其中攻击者从目标模型接收到的信息逐渐减少：置信向量（CV）披露，其中输出概率向量被发布；真实标签置信度（TLC）披露，其中只有模型分配给真实标签的概率是可用的；以及决策集（DS）披露。

    arXiv:2402.10686v1 Announce Type: cross  Abstract: In a membership inference attack (MIA), an attacker exploits the overconfidence exhibited by typical machine learning models to determine whether a specific data point was used to train a target model. In this paper, we analyze the performance of the state-of-the-art likelihood ratio attack (LiRA) within an information-theoretical framework that allows the investigation of the impact of the aleatoric uncertainty in the true data generation process, of the epistemic uncertainty caused by a limited training data set, and of the calibration level of the target model. We compare three different settings, in which the attacker receives decreasingly informative feedback from the target model: confidence vector (CV) disclosure, in which the output probability vector is released; true label confidence (TLC) disclosure, in which only the probability assigned to the true label is made available by the model; and decision set (DS) disclosure, in 
    
[^2]: 通过拓扑学习实现解缠编码

    Disentanglement Learning via Topology. (arXiv:2308.12696v1 [cs.LG])

    [http://arxiv.org/abs/2308.12696](http://arxiv.org/abs/2308.12696)

    本文提出了一种通过拓扑损失实现解缠编码的方法，这是第一个提出用于解缠的可微拓扑损失的论文，实验结果表明所提出的方法相对于最新结果改进了解缠得分。

    

    我们提出了TopDis（拓扑解缠），一种通过增加多尺度拓扑损失项学习解缠表示的方法。解缠是数据表示的关键属性，对深度学习模型的可解释性和鲁棒性以及高级认知的实现都非常重要。基于VAE的最新方法通过最小化潜变量的联合分布的总体相关性来实现解缠。我们从分析数据流形的拓扑属性的角度来看待解缠，特别是优化数据流形遍历的拓扑相似性。据我们所知，我们的论文是第一个提出用于解缠的可微拓扑损失的方法。我们的实验结果表明，所提出的拓扑损失相对于最新结果改进了解缠得分，如MIG、FactorVAE得分、SAP得分和DCI解缠得分。我们的方法以无监督的方式工作。

    We propose TopDis (Topological Disentanglement), a method for learning disentangled representations via adding multi-scale topological loss term. Disentanglement is a crucial property of data representations substantial for the explainability and robustness of deep learning models and a step towards high-level cognition. The state-of-the-art method based on VAE minimizes the total correlation of the joint distribution of latent variables. We take a different perspective on disentanglement by analyzing topological properties of data manifolds. In particular, we optimize the topological similarity for data manifolds traversals. To the best of our knowledge, our paper is the first one to propose a differentiable topological loss for disentanglement. Our experiments have shown that the proposed topological loss improves disentanglement scores such as MIG, FactorVAE score, SAP score and DCI disentanglement score with respect to state-of-the-art results. Our method works in an unsupervised m
    
[^3]: 鲁棒性AI生成文本检测的内部维度估计

    Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts. (arXiv:2306.04723v1 [cs.CL])

    [http://arxiv.org/abs/2306.04723](http://arxiv.org/abs/2306.04723)

    本文提出了衡量文本内部维度的方法，应用于鲁棒性AI生成文本的检测，展示了人类文本与AI生成文本在内部维度上的差异。

    

    快速提高的AI生成内容的质量使得很难区分人类和AI生成的文本，这可能会对社会产生不良影响。因此，研究人类文本的不变属性变得越来越重要。本文提出了一种人类文本的不变特征，即给定文本样本嵌入集合下的流形的内部维度。我们展示了自然语言流畅文本的平均内部维度在几个基于字母的语言中约为 $9$，而中文约为 $7$，而每种语言的AI生成文本的平均内部维度较低，差约 $1.5$，并且有明显的统计分离。

    Rapidly increasing quality of AI-generated content makes it difficult to distinguish between human and AI-generated texts, which may lead to undesirable consequences for society. Therefore, it becomes increasingly important to study the properties of human texts that are invariant over text domains and various proficiency of human writers, can be easily calculated for any language, and can robustly separate natural and AI-generated texts regardless of the generation model and sampling method. In this work, we propose such an invariant of human texts, namely the intrinsic dimensionality of the manifold underlying the set of embeddings of a given text sample. We show that the average intrinsic dimensionality of fluent texts in natural language is hovering around the value $9$ for several alphabet-based languages and around $7$ for Chinese, while the average intrinsic dimensionality of AI-generated texts for each language is $\approx 1.5$ lower, with a clear statistical separation between
    

