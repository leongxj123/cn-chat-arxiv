# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Instance-Optimal Cluster Recovery in the Labeled Stochastic Block Model.](http://arxiv.org/abs/2306.12968) | 本论文提出了一种算法，名为实例自适应聚类（IAC），它能够在标记随机块模型（LSBM）中恢复隐藏的群集。IAC包括一次谱聚类和一个迭代的基于似然的簇分配改进，不需要任何模型参数，是高效的。 |
| [^2] | [A Survey on Knowledge Graphs for Healthcare: Resources, Applications, and Promises.](http://arxiv.org/abs/2306.04802) | 本论文综述了医疗知识图谱(HKGs)的构建流程、关键技术和利用方法以及现有资源，并深入探讨了HKG在各种医疗领域的变革性影响。 |

# 详细

[^1]: 标记随机块模型中的最优簇恢复问题

    Instance-Optimal Cluster Recovery in the Labeled Stochastic Block Model. (arXiv:2306.12968v1 [cs.SI])

    [http://arxiv.org/abs/2306.12968](http://arxiv.org/abs/2306.12968)

    本论文提出了一种算法，名为实例自适应聚类（IAC），它能够在标记随机块模型（LSBM）中恢复隐藏的群集。IAC包括一次谱聚类和一个迭代的基于似然的簇分配改进，不需要任何模型参数，是高效的。

    

    本文考虑在有限数量的簇的情况下，用标记随机块模型（LSBM）恢复隐藏的社群，其中簇大小随着物品总数$n$的增长而线性增长。在LSBM中，为每对物品（独立地）观测到一个标签。我们的目标是设计一种有效的算法，利用观测到的标签来恢复簇。为此，我们重新审视了关于期望被任何聚类算法误分类的物品数量的实例特定下界。我们提出了实例自适应聚类（IAC），这是第一个在期望和高概率下都能匹配这些下界表现的算法。IAC由一次谱聚类算法和一个迭代的基于似然的簇分配改进组成。这种方法基于实例特定的下界，不需要任何模型参数，包括簇的数量。通过仅执行一次谱聚类，IAC在计算和存储方面都是高效的。

    We consider the problem of recovering hidden communities in the Labeled Stochastic Block Model (LSBM) with a finite number of clusters, where cluster sizes grow linearly with the total number $n$ of items. In the LSBM, a label is (independently) observed for each pair of items. Our objective is to devise an efficient algorithm that recovers clusters using the observed labels. To this end, we revisit instance-specific lower bounds on the expected number of misclassified items satisfied by any clustering algorithm. We present Instance-Adaptive Clustering (IAC), the first algorithm whose performance matches these lower bounds both in expectation and with high probability. IAC consists of a one-time spectral clustering algorithm followed by an iterative likelihood-based cluster assignment improvement. This approach is based on the instance-specific lower bound and does not require any model parameters, including the number of clusters. By performing the spectral clustering only once, IAC m
    
[^2]: 医疗知识图谱综述：资源、应用和前景

    A Survey on Knowledge Graphs for Healthcare: Resources, Applications, and Promises. (arXiv:2306.04802v1 [cs.AI])

    [http://arxiv.org/abs/2306.04802](http://arxiv.org/abs/2306.04802)

    本论文综述了医疗知识图谱(HKGs)的构建流程、关键技术和利用方法以及现有资源，并深入探讨了HKG在各种医疗领域的变革性影响。

    

    医疗知识图谱(HKGs)已成为组织医学知识的有结构且可解释的有为工具，提供了医学概念及其关系的全面视图。然而，数据异质性和覆盖范围有限等挑战仍然存在，强调了在HKG领域需要进一步研究的必要性。本综述是HKG的第一份综合概述。我们总结了HKG构建的流程和关键技术（即从头开始和通过集成），以及常见的利用方法（即基于模型和非基于模型）。为了为研究人员提供有价值的资源，我们根据它们捕获的数据类型和应用领域（该资源存储于https://github.com/lujiaying/Awesome-HealthCare-KnowledgeBase）组织了现有的HKG，并提供了相关的统计信息。在应用部分，我们深入探讨了HKG在各种医疗领域的变革性影响。

    Healthcare knowledge graphs (HKGs) have emerged as a promising tool for organizing medical knowledge in a structured and interpretable way, which provides a comprehensive view of medical concepts and their relationships. However, challenges such as data heterogeneity and limited coverage remain, emphasizing the need for further research in the field of HKGs. This survey paper serves as the first comprehensive overview of HKGs. We summarize the pipeline and key techniques for HKG construction (i.e., from scratch and through integration), as well as the common utilization approaches (i.e., model-free and model-based). To provide researchers with valuable resources, we organize existing HKGs (The resource is available at https://github.com/lujiaying/Awesome-HealthCare-KnowledgeBase) based on the data types they capture and application domains, supplemented with pertinent statistical information. In the application section, we delve into the transformative impact of HKGs across various hea
    

