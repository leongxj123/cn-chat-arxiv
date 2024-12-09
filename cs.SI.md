# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph Inference Acceleration by Learning MLPs on Graphs without Supervision](https://arxiv.org/abs/2402.08918) | 该论文提出了一个简单而有效的框架SimMLP，通过在图上无监督学习MLPs，提高了在延迟敏感的应用中的泛化能力。 |

# 详细

[^1]: 通过无监督在图上学习多层感知机（MLP）加速图推理

    Graph Inference Acceleration by Learning MLPs on Graphs without Supervision

    [https://arxiv.org/abs/2402.08918](https://arxiv.org/abs/2402.08918)

    该论文提出了一个简单而有效的框架SimMLP，通过在图上无监督学习MLPs，提高了在延迟敏感的应用中的泛化能力。

    

    图神经网络（GNNs）已经在各种图学习任务中展示出了有效性，但是它们对消息传递的依赖限制了它们在延迟敏感的应用中的部署，比如金融欺诈检测。最近的研究探索了从GNNs中提取知识到多层感知机（MLPs）来加速推理。然而，这种任务特定的有监督蒸馏限制了对未见节点的泛化，而在延迟敏感的应用中这种情况很常见。为此，我们提出了一种简单而有效的框架SimMLP，用于在图上无监督学习MLPs，以增强泛化能力。SimMLP利用自监督对齐GNNs和MLPs之间的节点特征和图结构之间的精细和泛化的相关性，并提出了两种策略来减轻平凡解的风险。从理论上讲，

    arXiv:2402.08918v1 Announce Type: cross Abstract: Graph Neural Networks (GNNs) have demonstrated effectiveness in various graph learning tasks, yet their reliance on message-passing constraints their deployment in latency-sensitive applications such as financial fraud detection. Recent works have explored distilling knowledge from GNNs to Multi-Layer Perceptrons (MLPs) to accelerate inference. However, this task-specific supervised distillation limits generalization to unseen nodes, which are prevalent in latency-sensitive applications. To this end, we present \textbf{\textsc{SimMLP}}, a \textbf{\textsc{Sim}}ple yet effective framework for learning \textbf{\textsc{MLP}}s on graphs without supervision, to enhance generalization. \textsc{SimMLP} employs self-supervised alignment between GNNs and MLPs to capture the fine-grained and generalizable correlation between node features and graph structures, and proposes two strategies to alleviate the risk of trivial solutions. Theoretically, w
    

