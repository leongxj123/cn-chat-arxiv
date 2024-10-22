# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improved prediction of ligand-protein binding affinities by meta-modeling.](http://arxiv.org/abs/2310.03946) | 通过整合基于结构的对接和基于序列的深度学习模型，开发了一个元模型框架，显著改善了配体-蛋白质结合亲和力预测的性能。 |
| [^2] | [Embed-Search-Align: DNA Sequence Alignment using Transformer Models.](http://arxiv.org/abs/2309.11087) | 这项研究使用Transformer模型对DNA序列进行对齐，通过生成数值表示来实现。相比传统方法，该方法在短DNA序列的分类任务上取得了更好的性能，对于基因组学分析具有潜在的应用价值。 |

# 详细

[^1]: 通过元模型改进了配体-蛋白质结合亲和力的预测

    Improved prediction of ligand-protein binding affinities by meta-modeling. (arXiv:2310.03946v1 [cs.LG])

    [http://arxiv.org/abs/2310.03946](http://arxiv.org/abs/2310.03946)

    通过整合基于结构的对接和基于序列的深度学习模型，开发了一个元模型框架，显著改善了配体-蛋白质结合亲和力预测的性能。

    

    通过计算方法准确筛选候选药物配体与靶蛋白的结合是药物开发的主要关注点，因为筛选潜在候选物能够节省找药物的时间和费用。这种虚拟筛选部分依赖于预测配体和蛋白质之间的结合亲和力的方法。鉴于存在许多计算模型对不同目标的结合亲和力预测结果不同，我们在这里开发了一个元模型框架，通过整合已发表的基于结构的对接和基于序列的深度学习模型来构建。在构建这个框架时，我们评估了许多组合的个别模型、训练数据库以及线性和非线性的元模型方法。我们显示出许多元模型在亲和力预测上显著改善了个别基础模型的性能。我们最好的元模型达到了与最先进的纯结构为基础的深度学习工具相当的性能。总体而言，我们证明了这个元模型框架可以显著改善配体-蛋白质结合亲和力预测的性能。

    The accurate screening of candidate drug ligands against target proteins through computational approaches is of prime interest to drug development efforts, as filtering potential candidates would save time and expenses for finding drugs. Such virtual screening depends in part on methods to predict the binding affinity between ligands and proteins. Given many computational models for binding affinity prediction with varying results across targets, we herein develop a meta-modeling framework by integrating published empirical structure-based docking and sequence-based deep learning models. In building this framework, we evaluate many combinations of individual models, training databases, and linear and nonlinear meta-modeling approaches. We show that many of our meta-models significantly improve affinity predictions over individual base models. Our best meta-models achieve comparable performance to state-of-the-art exclusively structure-based deep learning tools. Overall, we demonstrate 
    
[^2]: Embed-Search-Align: 使用Transformer模型进行DNA序列对齐

    Embed-Search-Align: DNA Sequence Alignment using Transformer Models. (arXiv:2309.11087v1 [q-bio.GN])

    [http://arxiv.org/abs/2309.11087](http://arxiv.org/abs/2309.11087)

    这项研究使用Transformer模型对DNA序列进行对齐，通过生成数值表示来实现。相比传统方法，该方法在短DNA序列的分类任务上取得了更好的性能，对于基因组学分析具有潜在的应用价值。

    

    DNA序列对齐涉及将短DNA读取分配到广泛的参考基因组上的最可能位置。这个过程对于各种基因组学分析至关重要，包括变异调用、转录组学和表观基因组学。传统方法经过数十年的改进，以两个步骤解决这个挑战：先进行基因组索引，然后进行高效搜索以确定给定读取的可能位置。在大规模语言模型（LLM）在将文本编码为嵌入向量方面取得成功的基础上，最近的研究努力探索了是否可以使用相同的Transformer架构为DNA序列生成数值表示。这样的模型已经在涉及分类短DNA序列的任务中显示出早期的潜力，例如检测编码和非编码区域以及识别增强子和启动子序列。然而，序列分类任务的性能并不能直接应用于序列对齐任务，对齐任务的关键是在保持序列相似性的同时找到最佳的对应位置。

    DNA sequence alignment involves assigning short DNA reads to the most probable locations on an extensive reference genome. This process is crucial for various genomic analyses, including variant calling, transcriptomics, and epigenomics. Conventional methods, refined over decades, tackle this challenge in two steps: genome indexing followed by efficient search to locate likely positions for given reads. Building on the success of Large Language Models (LLM) in encoding text into embeddings, where the distance metric captures semantic similarity, recent efforts have explored whether the same Transformer architecture can produce numerical representations for DNA sequences. Such models have shown early promise in tasks involving classification of short DNA sequences, such as the detection of coding vs non-coding regions, as well as the identification of enhancer and promoter sequences. Performance at sequence classification tasks does not, however, translate to sequence alignment, where i
    

