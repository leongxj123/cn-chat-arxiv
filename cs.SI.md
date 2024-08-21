# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inferring Dynamic Networks from Marginals with Iterative Proportional Fitting](https://arxiv.org/abs/2402.18697) | 通过识别一个生成网络模型，我们建立了一个设置，IPF可以恢复最大似然估计，揭示了关于在这种设置中使用IPF的隐含假设，并可以为IPF的参数估计提供结构相关的误差界。 |
| [^2] | [Finding Already Debunked Narratives via Multistage Retrieval: Enabling Cross-Lingual, Cross-Dataset and Zero-Shot Learning.](http://arxiv.org/abs/2308.05680) | 本研究通过创建新的数据集、评估多语言预训练Transformer模型以及提出多阶段框架来解决了跨语言澄清检索问题。 |

# 详细

[^1]: 从边际推断动态网络的方法：迭代比例拟合

    Inferring Dynamic Networks from Marginals with Iterative Proportional Fitting

    [https://arxiv.org/abs/2402.18697](https://arxiv.org/abs/2402.18697)

    通过识别一个生成网络模型，我们建立了一个设置，IPF可以恢复最大似然估计，揭示了关于在这种设置中使用IPF的隐含假设，并可以为IPF的参数估计提供结构相关的误差界。

    

    来自现实数据约束的常见网络推断问题是如何从时间聚合的邻接矩阵和时间变化边际（即行向量和列向量之和）推断动态网络。先前的方法为了解决这个问题重新利用了经典的迭代比例拟合（IPF）过程，也称为Sinkhorn算法，并取得了令人满意的经验结果。然而，使用IPF的统计基础尚未得到很好的理解：在什么情况下，IPF提供了从边际准确估计动态网络的原则性，以及它在多大程度上估计了网络？在这项工作中，我们确定了这样一个设置，通过识别一个生成网络模型，IPF可以恢复其最大似然估计。我们的模型揭示了关于在这种设置中使用IPF的隐含假设，并使得可以进行新的分析，如有关IPF参数估计的结构相关误差界。当IPF失败时

    arXiv:2402.18697v1 Announce Type: cross  Abstract: A common network inference problem, arising from real-world data constraints, is how to infer a dynamic network from its time-aggregated adjacency matrix and time-varying marginals (i.e., row and column sums). Prior approaches to this problem have repurposed the classic iterative proportional fitting (IPF) procedure, also known as Sinkhorn's algorithm, with promising empirical results. However, the statistical foundation for using IPF has not been well understood: under what settings does IPF provide principled estimation of a dynamic network from its marginals, and how well does it estimate the network? In this work, we establish such a setting, by identifying a generative network model whose maximum likelihood estimates are recovered by IPF. Our model both reveals implicit assumptions on the use of IPF in such settings and enables new analyses, such as structure-dependent error bounds on IPF's parameter estimates. When IPF fails to c
    
[^2]: 通过多阶段检索找到已经被澄清的叙述：实现跨语言、跨数据集和零样本学习

    Finding Already Debunked Narratives via Multistage Retrieval: Enabling Cross-Lingual, Cross-Dataset and Zero-Shot Learning. (arXiv:2308.05680v1 [cs.CL])

    [http://arxiv.org/abs/2308.05680](http://arxiv.org/abs/2308.05680)

    本研究通过创建新的数据集、评估多语言预训练Transformer模型以及提出多阶段框架来解决了跨语言澄清检索问题。

    

    检索已经被澄清的叙述的任务旨在检测已经经过事实核查的故事。成功检测到已被澄清的声明不仅减少了专业事实核查人员的手动努力，还可以有助于减缓虚假信息的传播。由于缺乏可用数据，这是一个研究不足的问题，特别是在考虑跨语言任务时，即在检查的在线帖子的语言与事实核查文章的语言不同的情况下进行检索。本文通过以下方式填补了这一空白：（i）创建了一个新颖的数据集，以允许对已被澄清的叙述进行跨语言检索的研究，使用推文作为对事实核查文章数据库的查询；（ii）展示了一个全面的实验，以评估经过微调和现成的多语言预训练Transformer模型在这个任务上的性能；（iii）提出了一个新颖的多阶段框架，将这个跨语言澄清检索问题划分为不同的阶段。

    The task of retrieving already debunked narratives aims to detect stories that have already been fact-checked. The successful detection of claims that have already been debunked not only reduces the manual efforts of professional fact-checkers but can also contribute to slowing the spread of misinformation. Mainly due to the lack of readily available data, this is an understudied problem, particularly when considering the cross-lingual task, i.e. the retrieval of fact-checking articles in a language different from the language of the online post being checked. This paper fills this gap by (i) creating a novel dataset to enable research on cross-lingual retrieval of already debunked narratives, using tweets as queries to a database of fact-checking articles; (ii) presenting an extensive experiment to benchmark fine-tuned and off-the-shelf multilingual pre-trained Transformer models for this task; and (iii) proposing a novel multistage framework that divides this cross-lingual debunk ret
    

