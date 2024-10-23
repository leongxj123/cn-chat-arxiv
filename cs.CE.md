# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DNABERT-S: Learning Species-Aware DNA Embedding with Genome Foundation Models](https://arxiv.org/abs/2402.08777) | DNABERT-S是一种专门用于创建物种感知的DNA嵌入的基因组基础模型。为了提高对长读DNA序列的嵌入效果，引入了Manifold Instance Mixup (MI-Mix)对比目标方法来训练模型。 |

# 详细

[^1]: DNABERT-S: 学习具有基因组基础模型的物种感知DNA嵌入

    DNABERT-S: Learning Species-Aware DNA Embedding with Genome Foundation Models

    [https://arxiv.org/abs/2402.08777](https://arxiv.org/abs/2402.08777)

    DNABERT-S是一种专门用于创建物种感知的DNA嵌入的基因组基础模型。为了提高对长读DNA序列的嵌入效果，引入了Manifold Instance Mixup (MI-Mix)对比目标方法来训练模型。

    

    有效的DNA嵌入在基因组分析中仍然至关重要，特别是在缺乏用于模型微调的标记数据的情况下，尽管基因组基础模型已经取得了显著进展。一个典型的例子是宏基因组分箱，这是微生物组研究中的一个关键过程，旨在通过来自可能包含成千上万个不同的、通常没有经过表征的物种的复杂混合DNA序列的物种来对DNA序列进行分组。为了填补有效的DNA嵌入模型的缺陷，我们引入了DNABERT-S，这是一个专门用于创建物种感知的DNA嵌入的基因组基础模型。为了鼓励对易出错的长读DNA序列进行有效嵌入，我们引入了Manifold Instance Mixup(MI-Mix)，一种对比目标，它在随机选择的层次中混合DNA序列的隐藏表示，并训练模型以在输出层识别和区分这些混合比例。

    arXiv:2402.08777v1 Announce Type: cross Abstract: Effective DNA embedding remains crucial in genomic analysis, particularly in scenarios lacking labeled data for model fine-tuning, despite the significant advancements in genome foundation models. A prime example is metagenomics binning, a critical process in microbiome research that aims to group DNA sequences by their species from a complex mixture of DNA sequences derived from potentially thousands of distinct, often uncharacterized species. To fill the lack of effective DNA embedding models, we introduce DNABERT-S, a genome foundation model that specializes in creating species-aware DNA embeddings. To encourage effective embeddings to error-prone long-read DNA sequences, we introduce Manifold Instance Mixup (MI-Mix), a contrastive objective that mixes the hidden representations of DNA sequences at randomly selected layers and trains the model to recognize and differentiate these mixed proportions at the output layer. We further enha
    

