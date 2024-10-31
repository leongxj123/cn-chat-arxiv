# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fair Coresets via Optimal Transport](https://arxiv.org/abs/2311.05436) | 本研究提出了公平的Wasserstein核心集(FWC)，该方法通过最小化原始数据集与加权合成样本之间的Wasserstein距离，并强制实现人口平等，生成公平的合成代表性样本，可用于下游学习任务。 |
| [^2] | [Fair Ranking under Disparate Uncertainty](https://arxiv.org/abs/2309.01610) | 提出了一种新的公平排名标准Equal-Opportunity Ranking（EOR），将底层相关性模型的不确定性差异考虑在内，通过组内公平抽奖实现公平排名。 |

# 详细

[^1]: 通过最优传输实现公平的核心集

    Fair Coresets via Optimal Transport

    [https://arxiv.org/abs/2311.05436](https://arxiv.org/abs/2311.05436)

    本研究提出了公平的Wasserstein核心集(FWC)，该方法通过最小化原始数据集与加权合成样本之间的Wasserstein距离，并强制实现人口平等，生成公平的合成代表性样本，可用于下游学习任务。

    

    数据精炼和核心集已成为生成用于处理大规模数据集的下游学习任务的较小代表性样本集的流行方法。与此同时，机器学习越来越多地应用于社会层面的决策过程，使得模型构建者必须解决存在于数据中的子群体的固有偏见问题。当前方法通过优化相对于原始样本的局部属性来创建公平的合成代表性样本，但其对下游学习过程的影响尚未被探索。在这项工作中，我们提出了公平的Wasserstein核心集（FWC），一种新颖的核心集方法，它生成既具有公平性的合成代表性样本，又具有用于下游学习任务的样本级权重。FWC最小化原始数据集与加权合成样本之间的Wasserstein距离，同时强制实现人口平等。我们展示了FWC的无约束版本等价于通常的最优传输问题，并且通过实验证明了FWC的有效性和公平性。

    Data distillation and coresets have emerged as popular approaches to generate a smaller representative set of samples for downstream learning tasks to handle large-scale datasets. At the same time, machine learning is being increasingly applied to decision-making processes at a societal level, making it imperative for modelers to address inherent biases towards subgroups present in the data. Current approaches create fair synthetic representative samples by optimizing local properties relative to the original samples, but their effect on downstream learning processes has yet to be explored. In this work, we present fair Wasserstein coresets (FWC), a novel coreset approach which generates fair synthetic representative samples along with sample-level weights to be used in downstream learning tasks. FWC minimizes the Wasserstein distance between the original dataset and the weighted synthetic samples while enforcing demographic parity. We show that an unconstrained version of FWC is equiv
    
[^2]: 不同不确定性下的公平排名

    Fair Ranking under Disparate Uncertainty

    [https://arxiv.org/abs/2309.01610](https://arxiv.org/abs/2309.01610)

    提出了一种新的公平排名标准Equal-Opportunity Ranking（EOR），将底层相关性模型的不确定性差异考虑在内，通过组内公平抽奖实现公平排名。

    

    排名是一种广泛使用的方法，用于将人类评估者的注意力集中在可管理的选项子集上。它作为人类决策过程的一部分的使用范围从在电子商务网站上展示潜在相关产品到为人工审查优先处理大学申请。虽然排名可以通过将关注集中在最有前途的选项上使人类评估更加高效，但我们认为，如果底层相关性模型的不确定性在不同组别的选项之间存在差异，排名可能会引入不公平。不幸的是，这种不确定性差异似乎普遍存在，常常对少数群体造成损害，因为这些群体的相关性估计可能由于缺乏数据或合适的特征而具有更高的不确定性。为了解决这个公平问题，我们提出了Equal-Opportunity Ranking（EOR）作为排名的新公平标准，并展示它对应于在相关选项之间进行组内公平抽奖

    arXiv:2309.01610v2 Announce Type: replace  Abstract: Ranking is a ubiquitous method for focusing the attention of human evaluators on a manageable subset of options. Its use as part of human decision-making processes ranges from surfacing potentially relevant products on an e-commerce site to prioritizing college applications for human review. While ranking can make human evaluation more effective by focusing attention on the most promising options, we argue that it can introduce unfairness if the uncertainty of the underlying relevance model differs between groups of options. Unfortunately, such disparity in uncertainty appears widespread, often to the detriment of minority groups for which relevance estimates can have higher uncertainty due to a lack of data or appropriate features. To address this fairness issue, we propose Equal-Opportunity Ranking (EOR) as a new fairness criterion for ranking and show that it corresponds to a group-wise fair lottery among the relevant options even
    

