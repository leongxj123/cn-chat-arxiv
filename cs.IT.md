# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TREET: TRansfer Entropy Estimation via Transformer](https://arxiv.org/abs/2402.06919) | 本研究提出了TREET，一种基于Transformer的传输熵估计方法，通过引入Donsker-Vardhan表示法和注意力机制，实现了对稳定过程的传输熵估计。我们设计了估计TE的优化方案，并展示了通过联合优化方案优化通信通道容量和估计器的记忆能力。 |

# 详细

[^1]: TREET: 基于Transformer的传输熵估计

    TREET: TRansfer Entropy Estimation via Transformer

    [https://arxiv.org/abs/2402.06919](https://arxiv.org/abs/2402.06919)

    本研究提出了TREET，一种基于Transformer的传输熵估计方法，通过引入Donsker-Vardhan表示法和注意力机制，实现了对稳定过程的传输熵估计。我们设计了估计TE的优化方案，并展示了通过联合优化方案优化通信通道容量和估计器的记忆能力。

    

    传输熵（TE）是信息论中揭示过程之间信息流动方向的度量，对各种实际应用提供了宝贵的见解。本研究提出了一种名为TREET的基于Transformer的传输熵估计方法，用于估计稳定过程的TE。所提出的方法利用Donsker-Vardhan（DV）表示法对TE进行估计，并利用注意力机制进行神经估计任务。我们对TREET进行了详细的理论和实证研究，并将其与现有方法进行了比较。为了增加其适用性，我们设计了一种基于功能表示引理的估计TE优化方案。之后，我们利用联合优化方案来优化具有记忆性的通信通道容量，这是信息论中的一个典型优化问题，并展示了我们估计器的记忆能力。

    Transfer entropy (TE) is a measurement in information theory that reveals the directional flow of information between processes, providing valuable insights for a wide range of real-world applications. This work proposes Transfer Entropy Estimation via Transformers (TREET), a novel transformer-based approach for estimating the TE for stationary processes. The proposed approach employs Donsker-Vardhan (DV) representation to TE and leverages the attention mechanism for the task of neural estimation. We propose a detailed theoretical and empirical study of the TREET, comparing it to existing methods. To increase its applicability, we design an estimated TE optimization scheme that is motivated by the functional representation lemma. Afterwards, we take advantage of the joint optimization scheme to optimize the capacity of communication channels with memory, which is a canonical optimization problem in information theory, and show the memory capabilities of our estimator. Finally, we apply
    

