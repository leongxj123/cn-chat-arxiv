# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Distributed Larger-Than-Memory Subset Selection With Pairwise Submodular Functions](https://arxiv.org/abs/2402.16442) | 本文提出了一种新颖的分布式约束算法，通过迭代绑定最小和最大效用值来选择高质量的点并丢弃不重要的点。 |
| [^2] | [Mitigating Biases with Diverse Ensembles and Diffusion Models](https://arxiv.org/abs/2311.16176) | 通过利用扩散概率模型（DPMs）生成新特征组合的图像，可以在集成模型中增加模型多样性，并减轻捷径偏见，而无需额外监督信号。 |

# 详细

[^1]: 在具有配对次模模函数的分布式大于内存的子集选择问题研究

    On Distributed Larger-Than-Memory Subset Selection With Pairwise Submodular Functions

    [https://arxiv.org/abs/2402.16442](https://arxiv.org/abs/2402.16442)

    本文提出了一种新颖的分布式约束算法，通过迭代绑定最小和最大效用值来选择高质量的点并丢弃不重要的点。

    

    许多学习问题取决于子集选择的基本问题，即确定一组重要和代表性的点。本文提出了一种具有可证估计近似保证的新颖分布式约束算法，它通过迭代绑定最小和最大效用值来选择高质量的点并丢弃不重要的点。

    arXiv:2402.16442v1 Announce Type: cross  Abstract: Many learning problems hinge on the fundamental problem of subset selection, i.e., identifying a subset of important and representative points. For example, selecting the most significant samples in ML training cannot only reduce training costs but also enhance model quality. Submodularity, a discrete analogue of convexity, is commonly used for solving subset selection problems. However, existing algorithms for optimizing submodular functions are sequential, and the prior distributed methods require at least one central machine to fit the target subset. In this paper, we relax the requirement of having a central machine for the target subset by proposing a novel distributed bounding algorithm with provable approximation guarantees. The algorithm iteratively bounds the minimum and maximum utility values to select high quality points and discard the unimportant ones. When bounding does not find the complete subset, we use a multi-round, 
    
[^2]: 通过多样化合成和扩散模型减轻偏见

    Mitigating Biases with Diverse Ensembles and Diffusion Models

    [https://arxiv.org/abs/2311.16176](https://arxiv.org/abs/2311.16176)

    通过利用扩散概率模型（DPMs）生成新特征组合的图像，可以在集成模型中增加模型多样性，并减轻捷径偏见，而无需额外监督信号。

    

    数据中的虚假相关性，即多个线索可以预测目标标签，常常导致一种称为捷径偏见的现象，即模型依赖于错误的、易学的线索，而忽略可靠的线索。在这项工作中，我们提出了一种利用扩散概率模型（DPMs）的集成多样化框架，用于减轻捷径偏见。我们展示了在特定的训练间隔中，DPMs可以生成具有新特征组合的图像，即使在显示相关输入特征的样本上进行训练。我们利用这一关键属性通过集成不一致性生成合成反事实来增加模型的多样性。我们展示了DPM引导的多样化足以消除对主要捷径线索的依赖，无需额外的监督信号。我们进一步在几个多样化目标上在实证上量化其有效性，并最终展示了改进的泛化性能。

    arXiv:2311.16176v2 Announce Type: replace-cross  Abstract: Spurious correlations in the data, where multiple cues are predictive of the target labels, often lead to a phenomenon known as shortcut bias, where a model relies on erroneous, easy-to-learn cues while ignoring reliable ones. In this work, we propose an ensemble diversification framework exploiting Diffusion Probabilistic Models (DPMs) for shortcut bias mitigation. We show that at particular training intervals, DPMs can generate images with novel feature combinations, even when trained on samples displaying correlated input features. We leverage this crucial property to generate synthetic counterfactuals to increase model diversity via ensemble disagreement. We show that DPM-guided diversification is sufficient to remove dependence on primary shortcut cues, without a need for additional supervised signals. We further empirically quantify its efficacy on several diversification objectives, and finally show improved generalizati
    

