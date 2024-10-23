# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Transfer Learning Bayesian Optimization to Design Competitor DNA Molecules for Use in Diagnostic Assays](https://arxiv.org/abs/2402.17704) | 通过将转移学习的代理模型与贝叶斯优化相结合，本文展示了如何通过在优化任务之间共享信息来减少实验的总数，并且演示了在设计用于扩增基因诊断测定的DNA竞争对手时实验数量的减少。 |
| [^2] | [DNABERT-S: Learning Species-Aware DNA Embedding with Genome Foundation Models](https://arxiv.org/abs/2402.08777) | DNABERT-S是一种专门用于创建物种感知的DNA嵌入的基因组基础模型。为了提高对长读DNA序列的嵌入效果，引入了Manifold Instance Mixup (MI-Mix)对比目标方法来训练模型。 |

# 详细

[^1]: 将贝叶斯优化应用于转移学习以设计用于诊断测定的竞争对手DNA分子

    Transfer Learning Bayesian Optimization to Design Competitor DNA Molecules for Use in Diagnostic Assays

    [https://arxiv.org/abs/2402.17704](https://arxiv.org/abs/2402.17704)

    通过将转移学习的代理模型与贝叶斯优化相结合，本文展示了如何通过在优化任务之间共享信息来减少实验的总数，并且演示了在设计用于扩增基因诊断测定的DNA竞争对手时实验数量的减少。

    

    随着工程生物分子设备的兴起，定制生物序列的需求不断增加。通常，为了特定应用需要制作许多类似的生物序列，这意味着需要进行大量甚至昂贵的实验来优化这些序列。本文提出了一个转移学习设计实验工作流程，使这种开发变得可行。通过将转移学习代理模型与贝叶斯优化相结合，我们展示了如何通过在优化任务之间共享信息来减少实验的总数。我们演示了使用用于扩增基因诊断测定中使用的DNA竞争对手开发数据来减少实验数量。我们使用交叉验证来比较不同转移学习模型的预测准确性，然后比较这些模型在单一目标和惩罚优化下的性能。

    arXiv:2402.17704v1 Announce Type: cross  Abstract: With the rise in engineered biomolecular devices, there is an increased need for tailor-made biological sequences. Often, many similar biological sequences need to be made for a specific application meaning numerous, sometimes prohibitively expensive, lab experiments are necessary for their optimization. This paper presents a transfer learning design of experiments workflow to make this development feasible. By combining a transfer learning surrogate model with Bayesian optimization, we show how the total number of experiments can be reduced by sharing information between optimization tasks. We demonstrate the reduction in the number of experiments using data from the development of DNA competitors for use in an amplification-based diagnostic assay. We use cross-validation to compare the predictive accuracy of different transfer learning models, and then compare the performance of the models for both single objective and penalized opti
    
[^2]: DNABERT-S: 学习具有基因组基础模型的物种感知DNA嵌入

    DNABERT-S: Learning Species-Aware DNA Embedding with Genome Foundation Models

    [https://arxiv.org/abs/2402.08777](https://arxiv.org/abs/2402.08777)

    DNABERT-S是一种专门用于创建物种感知的DNA嵌入的基因组基础模型。为了提高对长读DNA序列的嵌入效果，引入了Manifold Instance Mixup (MI-Mix)对比目标方法来训练模型。

    

    有效的DNA嵌入在基因组分析中仍然至关重要，特别是在缺乏用于模型微调的标记数据的情况下，尽管基因组基础模型已经取得了显著进展。一个典型的例子是宏基因组分箱，这是微生物组研究中的一个关键过程，旨在通过来自可能包含成千上万个不同的、通常没有经过表征的物种的复杂混合DNA序列的物种来对DNA序列进行分组。为了填补有效的DNA嵌入模型的缺陷，我们引入了DNABERT-S，这是一个专门用于创建物种感知的DNA嵌入的基因组基础模型。为了鼓励对易出错的长读DNA序列进行有效嵌入，我们引入了Manifold Instance Mixup(MI-Mix)，一种对比目标，它在随机选择的层次中混合DNA序列的隐藏表示，并训练模型以在输出层识别和区分这些混合比例。

    arXiv:2402.08777v1 Announce Type: cross Abstract: Effective DNA embedding remains crucial in genomic analysis, particularly in scenarios lacking labeled data for model fine-tuning, despite the significant advancements in genome foundation models. A prime example is metagenomics binning, a critical process in microbiome research that aims to group DNA sequences by their species from a complex mixture of DNA sequences derived from potentially thousands of distinct, often uncharacterized species. To fill the lack of effective DNA embedding models, we introduce DNABERT-S, a genome foundation model that specializes in creating species-aware DNA embeddings. To encourage effective embeddings to error-prone long-read DNA sequences, we introduce Manifold Instance Mixup (MI-Mix), a contrastive objective that mixes the hidden representations of DNA sequences at randomly selected layers and trains the model to recognize and differentiate these mixed proportions at the output layer. We further enha
    

