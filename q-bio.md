# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [EvolMPNN: Predicting Mutational Effect on Homologous Proteins by Evolution Encoding](https://arxiv.org/abs/2402.13418) | EvolMPNN通过进化感知的方式捕捉蛋白质突变对于锚定蛋白质的影响，并最终生成综合蛋白质嵌入。 |

# 详细

[^1]: EvolMPNN：通过进化编码预测同源蛋白质的突变效应

    EvolMPNN: Predicting Mutational Effect on Homologous Proteins by Evolution Encoding

    [https://arxiv.org/abs/2402.13418](https://arxiv.org/abs/2402.13418)

    EvolMPNN通过进化感知的方式捕捉蛋白质突变对于锚定蛋白质的影响，并最终生成综合蛋白质嵌入。

    

    预测蛋白质属性对生物和医学进步至关重要。当前的蛋白工程通过对典型蛋白质（称为野生型）进行突变，构建同源蛋白质家族并研究其属性。然而，现有方法很容易忽略细微的突变，无法捕捉蛋白质属性的影响。为此，我们提出了EvolMPNN，一种具有进化感知的消息传递神经网络，用于学习进化感知的蛋白质嵌入。

    arXiv:2402.13418v1 Announce Type: new  Abstract: Predicting protein properties is paramount for biological and medical advancements. Current protein engineering mutates on a typical protein, called the wild-type, to construct a family of homologous proteins and study their properties. Yet, existing methods easily neglect subtle mutations, failing to capture the effect on the protein properties. To this end, we propose EvolMPNN, Evolution-aware Message Passing Neural Network, to learn evolution-aware protein embeddings. EvolMPNN samples sets of anchor proteins, computes evolutionary information by means of residues and employs a differentiable evolution-aware aggregation scheme over these sampled anchors. This way EvolMPNNcan capture the mutation effect on proteins with respect to the anchor proteins. Afterwards, the aggregated evolution-aware embeddings are integrated with sequence embeddings to generate final comprehensive protein embeddings. Our model shows up to 6.4% better than sta
    

