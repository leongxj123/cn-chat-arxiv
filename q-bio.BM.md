# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Importance Weighted Expectation-Maximization for Protein Sequence Design.](http://arxiv.org/abs/2305.00386) | 本文提出了一种名为IsEM-Pro的方法，用于根据给定适应性标准生成蛋白质序列。在推理期间，从其潜在空间采样可以增加多样性，指导了探索高适应性区域。实验表明，相比先前最佳方法，IsEM-Pro的平均适应性得分至少高出55％，并生成了更多样化和新颖的蛋白质序列。 |

# 详细

[^1]: 蛋白质序列设计的重要性加权期望最大化方法

    Importance Weighted Expectation-Maximization for Protein Sequence Design. (arXiv:2305.00386v1 [q-bio.BM])

    [http://arxiv.org/abs/2305.00386](http://arxiv.org/abs/2305.00386)

    本文提出了一种名为IsEM-Pro的方法，用于根据给定适应性标准生成蛋白质序列。在推理期间，从其潜在空间采样可以增加多样性，指导了探索高适应性区域。实验表明，相比先前最佳方法，IsEM-Pro的平均适应性得分至少高出55％，并生成了更多样化和新颖的蛋白质序列。

    

    在生物和化学领域，设计具有所需生物功能的蛋白质序列非常重要。最近的机器学习方法使用代理序列-功能模型替代昂贵的湿实验验证。本文提出了一种名为IsEM-Pro的方法，用于根据给定的适应性标准生成蛋白质序列。它是一个潜在的生成模型，并受到另外一个学习的马尔可夫随机场结构特征的增强。研究者使用蒙特卡罗期望最大化方法（MCEM）来学习这个模型。在推理期间，从其潜在空间采样可以增加多样性，而其MRF特征则指导了探索高适应性区域。在八项蛋白质序列设计任务中的实验表明，我们的IsEM-Pro的平均适应性得分至少比先前最佳方法高55％，并且生成了更多样化和新颖的蛋白质序列。

    Designing protein sequences with desired biological function is crucial in biology and chemistry. Recent machine learning methods use a surrogate sequence-function model to replace the expensive wet-lab validation. How can we efficiently generate diverse and novel protein sequences with high fitness? In this paper, we propose IsEM-Pro, an approach to generate protein sequences towards a given fitness criterion. At its core, IsEM-Pro is a latent generative model, augmented by combinatorial structure features from a separately learned Markov random fields (MRFs). We develop an Monte Carlo Expectation-Maximization method (MCEM) to learn the model. During inference, sampling from its latent space enhances diversity while its MRFs features guide the exploration in high fitness regions. Experiments on eight protein sequence design tasks show that our IsEM-Pro outperforms the previous best methods by at least 55% on average fitness score and generates more diverse and novel protein sequences.
    

