# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TRIALSCOPE A Unifying Causal Framework for Scaling Real-World Evidence Generation with Biomedical Language Models.](http://arxiv.org/abs/2311.01301) | TRIALSCOPE是一个统一的框架，利用生物医学语言模型将临床文本进行结构化，采用概率建模进行去噪和插补，并应用因果推断技术来应对混杂因素，以从实际世界数据中提取实证证据和推理临床假设。 |
| [^2] | [A Consistent and Scalable Algorithm for Best Subset Selection in Single Index Models.](http://arxiv.org/abs/2309.06230) | 该论文提出了针对高维单指数模型中最佳子集选择的一致性和可扩展算法，通过使用广义信息准则来确定支持的回归系数大小，消除了模型选择的调优需求，并具有子集选择一致性和高概率下的理想属性。 |

# 详细

[^1]: TRIALSCOPE：一个统一的因果框架，用于利用生物医学语言模型扩展实际世界证据生成

    TRIALSCOPE A Unifying Causal Framework for Scaling Real-World Evidence Generation with Biomedical Language Models. (arXiv:2311.01301v1 [cs.LG])

    [http://arxiv.org/abs/2311.01301](http://arxiv.org/abs/2311.01301)

    TRIALSCOPE是一个统一的框架，利用生物医学语言模型将临床文本进行结构化，采用概率建模进行去噪和插补，并应用因果推断技术来应对混杂因素，以从实际世界数据中提取实证证据和推理临床假设。

    

    实际世界数据的快速数字化为优化医疗服务和加速生物医学发现提供了前所未有的机会。然而，在实践中，这些数据往往以非结构化形式存在，如电子医疗记录中的临床笔记，并且通常受到混杂因素的困扰。本文介绍了TRIALSCOPE，一个用于从人群级观察数据中提取实际世界证据的统一框架。TRIALSCOPE利用生物医学语言模型来扩展规模化的临床文本，采用先进的概率建模进行去噪和插补，并结合最先进的因果推断技术来应对常见的混杂因素。利用临床试验规范作为通用表示形式，TRIALSCOPE提供了一个一键式解决方案，可使用观察数据生成和推理临床假设。在一个包含超过一百万个癌症患者的大规模实际世界数据集上进行了广泛的实验和分析。

    The rapid digitization of real-world data offers an unprecedented opportunity for optimizing healthcare delivery and accelerating biomedical discovery. In practice, however, such data is most abundantly available in unstructured forms, such as clinical notes in electronic medical records (EMRs), and it is generally plagued by confounders. In this paper, we present TRIALSCOPE, a unifying framework for distilling real-world evidence from population-level observational data. TRIALSCOPE leverages biomedical language models to structure clinical text at scale, employs advanced probabilistic modeling for denoising and imputation, and incorporates state-of-the-art causal inference techniques to combat common confounders. Using clinical trial specification as generic representation, TRIALSCOPE provides a turn-key solution to generate and reason with clinical hypotheses using observational data. In extensive experiments and analyses on a large-scale real-world dataset with over one million canc
    
[^2]: 单指数模型中最佳子集选择的一致性和可扩展算法

    A Consistent and Scalable Algorithm for Best Subset Selection in Single Index Models. (arXiv:2309.06230v1 [stat.ML])

    [http://arxiv.org/abs/2309.06230](http://arxiv.org/abs/2309.06230)

    该论文提出了针对高维单指数模型中最佳子集选择的一致性和可扩展算法，通过使用广义信息准则来确定支持的回归系数大小，消除了模型选择的调优需求，并具有子集选择一致性和高概率下的理想属性。

    

    高维数据的分析引发了对单指数模型（SIMs）和最佳子集选择的增加兴趣。SIMs为高维数据提供了一种可解释和灵活的建模框架，而最佳子集选择旨在从大量的预测因子中找到稀疏模型。然而，在高维模型中的最佳子集选择被认为是计算上难以处理的。现有的方法倾向于放宽选择，但不能得到最佳子集解。在本文中，我们通过提出第一个经过证明的针对高维SIMs中最佳子集选择的可扩展算法，直接解决了计算难题。我们的算法解具有子集选择一致性，并且几乎肯定具有用于参数估计的虚拟属性。该算法包括一个广义信息准则来确定回归系数的支持大小，消除模型选择调整。此外，我们的方法不假设误差分布或特定参数。

    Analysis of high-dimensional data has led to increased interest in both single index models (SIMs) and best subset selection. SIMs provide an interpretable and flexible modeling framework for high-dimensional data, while best subset selection aims to find a sparse model from a large set of predictors. However, best subset selection in high-dimensional models is known to be computationally intractable. Existing methods tend to relax the selection, but do not yield the best subset solution. In this paper, we directly tackle the intractability by proposing the first provably scalable algorithm for best subset selection in high-dimensional SIMs. Our algorithmic solution enjoys the subset selection consistency and has the oracle property with a high probability. The algorithm comprises a generalized information criterion to determine the support size of the regression coefficients, eliminating the model selection tuning. Moreover, our method does not assume an error distribution or a specif
    

