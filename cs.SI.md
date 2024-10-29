# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph Inference Acceleration by Learning MLPs on Graphs without Supervision](https://arxiv.org/abs/2402.08918) | 该论文提出了一个简单而有效的框架SimMLP，通过在图上无监督学习MLPs，提高了在延迟敏感的应用中的泛化能力。 |
| [^2] | [Multiscale Hodge Scattering Networks for Data Analysis](https://arxiv.org/abs/2311.10270) | 提出了多尺度霍奇散射网络（MHSNs），利用多尺度基础词典和卷积结构，生成对节点排列不变的特征。 |

# 详细

[^1]: 通过无监督在图上学习多层感知机（MLP）加速图推理

    Graph Inference Acceleration by Learning MLPs on Graphs without Supervision

    [https://arxiv.org/abs/2402.08918](https://arxiv.org/abs/2402.08918)

    该论文提出了一个简单而有效的框架SimMLP，通过在图上无监督学习MLPs，提高了在延迟敏感的应用中的泛化能力。

    

    图神经网络（GNNs）已经在各种图学习任务中展示出了有效性，但是它们对消息传递的依赖限制了它们在延迟敏感的应用中的部署，比如金融欺诈检测。最近的研究探索了从GNNs中提取知识到多层感知机（MLPs）来加速推理。然而，这种任务特定的有监督蒸馏限制了对未见节点的泛化，而在延迟敏感的应用中这种情况很常见。为此，我们提出了一种简单而有效的框架SimMLP，用于在图上无监督学习MLPs，以增强泛化能力。SimMLP利用自监督对齐GNNs和MLPs之间的节点特征和图结构之间的精细和泛化的相关性，并提出了两种策略来减轻平凡解的风险。从理论上讲，

    arXiv:2402.08918v1 Announce Type: cross Abstract: Graph Neural Networks (GNNs) have demonstrated effectiveness in various graph learning tasks, yet their reliance on message-passing constraints their deployment in latency-sensitive applications such as financial fraud detection. Recent works have explored distilling knowledge from GNNs to Multi-Layer Perceptrons (MLPs) to accelerate inference. However, this task-specific supervised distillation limits generalization to unseen nodes, which are prevalent in latency-sensitive applications. To this end, we present \textbf{\textsc{SimMLP}}, a \textbf{\textsc{Sim}}ple yet effective framework for learning \textbf{\textsc{MLP}}s on graphs without supervision, to enhance generalization. \textsc{SimMLP} employs self-supervised alignment between GNNs and MLPs to capture the fine-grained and generalizable correlation between node features and graph structures, and proposes two strategies to alleviate the risk of trivial solutions. Theoretically, w
    
[^2]: 用于数据分析的多尺度霍奇散射网络

    Multiscale Hodge Scattering Networks for Data Analysis

    [https://arxiv.org/abs/2311.10270](https://arxiv.org/abs/2311.10270)

    提出了多尺度霍奇散射网络（MHSNs），利用多尺度基础词典和卷积结构，生成对节点排列不变的特征。

    

    我们提出了一种新的散射网络，用于在单纯复合仿射上测量的信号，称为\emph{多尺度霍奇散射网络}（MHSNs）。我们的构造基于单纯复合仿射上的多尺度基础词典，即$\kappa$-GHWT和$\kappa$-HGLET，我们最近为给定单纯复合仿射中的维度$\kappa \in \mathbb{N}$推广了基于节点的广义哈-沃什变换（GHWT）和分层图拉普拉斯特征变换（HGLET）。$\kappa$-GHWT和$\kappa$-HGLET都形成冗余集合（即词典）的多尺度基础向量和给定信号的相应扩展系数。我们的MHSNs使用类似于卷积神经网络（CNN）的分层结构来级联词典系数模的矩。所得特征对单纯复合仿射的重新排序不变（即节点排列的置换

    arXiv:2311.10270v2 Announce Type: replace  Abstract: We propose new scattering networks for signals measured on simplicial complexes, which we call \emph{Multiscale Hodge Scattering Networks} (MHSNs). Our construction is based on multiscale basis dictionaries on simplicial complexes, i.e., the $\kappa$-GHWT and $\kappa$-HGLET, which we recently developed for simplices of dimension $\kappa \in \mathbb{N}$ in a given simplicial complex by generalizing the node-based Generalized Haar-Walsh Transform (GHWT) and Hierarchical Graph Laplacian Eigen Transform (HGLET). The $\kappa$-GHWT and the $\kappa$-HGLET both form redundant sets (i.e., dictionaries) of multiscale basis vectors and the corresponding expansion coefficients of a given signal. Our MHSNs use a layered structure analogous to a convolutional neural network (CNN) to cascade the moments of the modulus of the dictionary coefficients. The resulting features are invariant to reordering of the simplices (i.e., node permutation of the u
    

