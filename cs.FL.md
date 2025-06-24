# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [$L^*LM$: Learning Automata from Examples using Natural Language Oracles](https://arxiv.org/abs/2402.07051) | 该论文提出了一个名为 $L^*LM$ 的算法，通过自然语言和演示学习 DFA，提高了数据效率，具备强大的少样本学习能力。 |

# 详细

[^1]: $L^*LM$: 通过自然语言定义示例学习自动机

    $L^*LM$: Learning Automata from Examples using Natural Language Oracles

    [https://arxiv.org/abs/2402.07051](https://arxiv.org/abs/2402.07051)

    该论文提出了一个名为 $L^*LM$ 的算法，通过自然语言和演示学习 DFA，提高了数据效率，具备强大的少样本学习能力。

    

    专家演示已被证明是简化间接指定复杂任务的一种方法。最近的算法甚至支持从演示中提取明确的形式规范，如确定性有限自动机（DFA）。不幸的是，这些技术通常不具备高样本效率。在本文中，我们介绍了一种名为 $L^*LM$ 的算法，用于从演示和自然语言中学习 DFA。由于自然语言的表达能力，我们观察到从专家演示中学习 DFA 的数据效率显著提高。从技术上讲，$L^*LM$ 利用大型语言模型来回答关于底层任务的成员查询。然后将其与最近的演示学习技术相结合，将学习转化为一系列带标签示例学习问题。在我们的实验中，我们观察到这两种模态相互补充，从而产生了一个强大的少样本学习器。

    Expert demonstrations have proven an easy way to indirectly specify complex tasks. Recent algorithms even support extracting unambiguous formal specifications, e.g. deterministic finite automata (DFA), from demonstrations. Unfortunately, these techniques are generally not sample efficient. In this work, we introduce $L^*LM$, an algorithm for learning DFAs from both demonstrations and natural language. Due to the expressivity of natural language, we observe a significant improvement in the data efficiency of learning DFAs from expert demonstrations. Technically, $L^*LM$ leverages large language models to answer membership queries about the underlying task. This is then combined with recent techniques for transforming learning from demonstrations into a sequence of labeled example learning problems. In our experiments, we observe the two modalities complement each other, yielding a powerful few-shot learner.
    

