# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [KTbench: A Novel Data Leakage-Free Framework for Knowledge Tracing](https://arxiv.org/abs/2403.15304) | KTbench提出了一种无数据泄漏的知识追踪框架，解决了KT模型中KC之间的相关性学习可能导致的性能下降问题。 |

# 详细

[^1]: KTbench：一种全新的无数据泄漏的知识追踪框架

    KTbench: A Novel Data Leakage-Free Framework for Knowledge Tracing

    [https://arxiv.org/abs/2403.15304](https://arxiv.org/abs/2403.15304)

    KTbench提出了一种无数据泄漏的知识追踪框架，解决了KT模型中KC之间的相关性学习可能导致的性能下降问题。

    

    知识追踪（KT）涉及在智能辅导系统中预测学生对学习项目的未来表现。学习项目被标记为称为知识概念（KCs）的技能标签。许多KT模型通过用构成KC的学习项目取代学习项目来将学习项目-学生交互序列扩展为KC-学生交互序列，从而解决了稀疏的学习项目-学生交互问题并最小化了模型参数。然而，这种方法存在两个问题。第一个问题是模型学习同一项目内的KC之间的相关性的能力，这可能导致基本事实标签的泄漏并阻碍模型性能。第二个问题是现有的基准实现忽略了计数问题

    arXiv:2403.15304v1 Announce Type: cross  Abstract: Knowledge Tracing (KT) is concerned with predicting students' future performance on learning items in intelligent tutoring systems. Learning items are tagged with skill labels called knowledge concepts (KCs). Many KT models expand the sequence of item-student interactions into KC-student interactions by replacing learning items with their constituting KCs. This often results in a longer sequence length. This approach addresses the issue of sparse item-student interactions and minimises model parameters. However, two problems have been identified with such models.   The first problem is the model's ability to learn correlations between KCs belonging to the same item, which can result in the leakage of ground truth labels and hinder performance. This problem can lead to a significant decrease in performance on datasets with a higher number of KCs per item. The second problem is that the available benchmark implementations ignore accounti
    

