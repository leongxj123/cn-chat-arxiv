# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Few-Shot Learning on Graphs: from Meta-learning to Pre-training and Prompting](https://rss.arxiv.org/abs/2402.01440) | 本文综述了图上的小样本学习的最新发展，将现有的研究方法划分为元学习、预训练和混合方法三大类别，并对它们的优缺点进行了比较。还提出了未来的研究方向。 |
| [^2] | [MQuinE: a cure for "Z-paradox'' in knowledge graph embedding models](https://arxiv.org/abs/2402.03583) | 研究者发现知识图谱嵌入模型存在的“Z-悖论”限制了其表达能力，并提出了一种名为MQuinE的新模型，通过理论证明，MQuinE成功解决了Z-悖论，并在链接预测任务中显著优于现有模型。 |

# 详细

[^1]: 在图上的小样本学习：从元学习到预训练和提示

    Few-Shot Learning on Graphs: from Meta-learning to Pre-training and Prompting

    [https://rss.arxiv.org/abs/2402.01440](https://rss.arxiv.org/abs/2402.01440)

    本文综述了图上的小样本学习的最新发展，将现有的研究方法划分为元学习、预训练和混合方法三大类别，并对它们的优缺点进行了比较。还提出了未来的研究方向。

    

    图表示学习是图中心任务中的关键步骤，在这方面已经取得了重大进展。早期的技术通常在端到端的设置中运行，性能严重依赖于充足的标记数据的可用性。这个限制引发了图上的小样本学习的出现，其中每个任务只有少量的任务特定标签可用。鉴于这个领域的广泛文献，本综述试图综合最近的发展，提供比较性的见解，并确定未来的方向。我们将现有的研究系统地分为三个主要类别：元学习方法、预训练方法和混合方法，并在每个类别中进行细粒度的分类，以帮助读者进行方法选择。在每个类别中，我们分析这些方法之间的关系并比较它们的优缺点。最后，我们概述了图上的小样本学习未来的方向。

    Graph representation learning, a critical step in graph-centric tasks, has seen significant advancements. Earlier techniques often operate in an end-to-end setting, where performance heavily relies on the availability of ample labeled data. This constraint has spurred the emergence of few-shot learning on graphs, where only a few task-specific labels are available for each task. Given the extensive literature in this field, this survey endeavors to synthesize recent developments, provide comparative insights, and identify future directions. We systematically categorize existing studies into three major families: meta-learning approaches, pre-training approaches, and hybrid approaches, with a finer-grained classification in each family to aid readers in their method selection process. Within each category, we analyze the relationships among these methods and compare their strengths and limitations. Finally, we outline prospective future directions for few-shot learning on graphs to cata
    
[^2]: MQuinE:知识图谱嵌入模型中“Z-悖论”的解决方案

    MQuinE: a cure for "Z-paradox'' in knowledge graph embedding models

    [https://arxiv.org/abs/2402.03583](https://arxiv.org/abs/2402.03583)

    研究者发现知识图谱嵌入模型存在的“Z-悖论”限制了其表达能力，并提出了一种名为MQuinE的新模型，通过理论证明，MQuinE成功解决了Z-悖论，并在链接预测任务中显著优于现有模型。

    

    知识图谱嵌入（KGE）模型在许多知识图谱任务，包括链接预测和信息检索方面取得了最先进的结果。尽管KGE模型在实践中表现出优越性能，但我们发现一些流行的现有KGE模型存在表达不足的问题，称为“Z-悖论”。受到Z-悖论的存在的启发，我们提出了一种新的KGE模型，称为MQuinE，在不受Z-悖论的困扰的同时，保持强大的表达能力来模拟各种关系模式，包括对称/非对称，逆向，1-N/N-1/N-N和组合关系，并提供了理论上的证明。对实际知识库的实验表明，Z-悖论确实降低了现有KGE模型的性能，并且可能导致某些具有挑战性的测试样本的准确性下降超过20％。我们的实验进一步证明了MQuinE可以减轻Z-悖论的负面影响，并在链接预测方面以明显优势超越现有的KGE模型。

    Knowledge graph embedding (KGE) models achieved state-of-the-art results on many knowledge graph tasks including link prediction and information retrieval. Despite the superior performance of KGE models in practice, we discover a deficiency in the expressiveness of some popular existing KGE models called \emph{Z-paradox}. Motivated by the existence of Z-paradox, we propose a new KGE model called \emph{MQuinE} that does not suffer from Z-paradox while preserves strong expressiveness to model various relation patterns including symmetric/asymmetric, inverse, 1-N/N-1/N-N, and composition relations with theoretical justification. Experiments on real-world knowledge bases indicate that Z-paradox indeed degrades the performance of existing KGE models, and can cause more than 20\% accuracy drop on some challenging test samples. Our experiments further demonstrate that MQuinE can mitigate the negative impact of Z-paradox and outperform existing KGE models by a visible margin on link prediction
    

