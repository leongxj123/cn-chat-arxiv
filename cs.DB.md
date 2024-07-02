# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [$R^3$-NL2GQL: A Hybrid Models Approach for for Accuracy Enhancing and Hallucinations Mitigation.](http://arxiv.org/abs/2311.01862) | $R^3$-NL2GQL是一种通过利用较小和较大的Foundation Models进行重新排名、重写和细化的方法，以提高准确性和减轻幻觉，解决了NL2GQL任务中GQL生成能力和跨模式通用能力的挑战。 |
| [^2] | [Autumn: A Scalable Read Optimized LSM-tree based Key-Value Stores with Fast Point and Range Read Speed.](http://arxiv.org/abs/2305.05074) | Autumn是一个可扩展的、面向读操作优化的LSM-tree键值存储引擎，其创新之处在于通过动态调整相邻两层之间的容量比来不断提高读性能，使得点读和区间读成本从之前最优的$O(logN)$复杂度优化到了$O(\sqrt{logN})$。 |

# 详细

[^1]: $R^3$-NL2GQL:一种用于提高准确性和减轻幻觉的混合模型方法

    $R^3$-NL2GQL: A Hybrid Models Approach for for Accuracy Enhancing and Hallucinations Mitigation. (arXiv:2311.01862v1 [cs.CL])

    [http://arxiv.org/abs/2311.01862](http://arxiv.org/abs/2311.01862)

    $R^3$-NL2GQL是一种通过利用较小和较大的Foundation Models进行重新排名、重写和细化的方法，以提高准确性和减轻幻觉，解决了NL2GQL任务中GQL生成能力和跨模式通用能力的挑战。

    

    当前使用Foundation Models构建的NL2SQL任务取得了令人称赞的结果，然而直接将其应用于自然语言到图查询语言（NL2GQL）任务面临挑战，原因是GQL和SQL表达式之间存在显著差异，且GQL存在多种类型。我们的实验表明，在NL2GQL任务中，更大的Foundation Models展示了优越的跨模式通用能力，而较小的Foundation Models则通过微调难以提高其GQL生成能力。然而，在微调后，较小的模型表现出更好的意图理解和更高的语法准确性。与基于规则和槽填充技术不同，我们引入了R3-NL2GQL，该方法将较小和较大的Foundation Models用作重新排名、重写和细化器。该方法利用较小模型的理解能力进行信息的重新排名和重写，并利用卓越的通用化和生成能力进行细化。

    While current NL2SQL tasks constructed using Foundation Models have achieved commendable results, their direct application to Natural Language to Graph Query Language (NL2GQL) tasks poses challenges due to the significant differences between GQL and SQL expressions, as well as the numerous types of GQL. Our extensive experiments reveal that in NL2GQL tasks, larger Foundation Models demonstrate superior cross-schema generalization abilities, while smaller Foundation Models struggle to improve their GQL generation capabilities through fine-tuning. However, after fine-tuning, smaller models exhibit better intent comprehension and higher grammatical accuracy. Diverging from rule-based and slot-filling techniques, we introduce R3-NL2GQL, which employs both smaller and larger Foundation Models as reranker, rewriter and refiner. The approach harnesses the comprehension ability of smaller models for information reranker and rewriter, and the exceptional generalization and generation capabiliti
    
[^2]: Autumn：基于LSM-tree的可扩展的面向读操作优化的键值存储引擎

    Autumn: A Scalable Read Optimized LSM-tree based Key-Value Stores with Fast Point and Range Read Speed. (arXiv:2305.05074v1 [cs.DB])

    [http://arxiv.org/abs/2305.05074](http://arxiv.org/abs/2305.05074)

    Autumn是一个可扩展的、面向读操作优化的LSM-tree键值存储引擎，其创新之处在于通过动态调整相邻两层之间的容量比来不断提高读性能，使得点读和区间读成本从之前最优的$O(logN)$复杂度优化到了$O(\sqrt{logN})$。

    

    基于Log Structured Merge Trees (LSM-tree)的键值存储引擎被广泛应用于许多存储系统中，以支持更新、点读和区间读等各种操作。本文中，我们提出了一个名为Autumn的可扩展的、面向读操作优化的基于LSM-tree的键值存储引擎，它具有最少的点读和区间读成本。通过动态调整相邻两层之间的容量比来不断提高读性能，点读和区间读成本从之前最优的$O(logN)$复杂度优化到了$O(\sqrt{logN})$，并应用了新的Garnering合并策略。Autumn是一个可扩展的、面向读操作优化的LSM-tree键值存储引擎。

    The Log Structured Merge Trees (LSM-tree) based key-value stores are widely used in many storage systems to support a variety of operations such as updates, point reads, and range reads. Traditionally, LSM-tree's merge policy organizes data into multiple levels of exponentially increasing capacity to support high-speed writes. However, we contend that the traditional merge policies are not optimized for reads. In this work, we present Autumn, a scalable and read optimized LSM-tree based key-value stores with minimal point and range read cost. The key idea in improving the read performance is to dynamically adjust the capacity ratio between two adjacent levels as more data are stored. As a result, smaller levels gradually increase their capacities and merge more often. In particular, the point and range read cost improves from the previous best known $O(logN)$ complexity to $O(\sqrt{logN})$ in Autumn by applying the new novel Garnering merge policy. While Garnering merge policy optimize
    

