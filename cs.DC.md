# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DFML: Decentralized Federated Mutual Learning](https://arxiv.org/abs/2402.01863) | DFML是一个无服务器的分散式联邦互联学习框架，能够有效地处理模型和数据的异质性，并通过相互学习在客户端之间传授知识，以获得更快的收敛速度和更高的全局准确性。 |

# 详细

[^1]: DFML：分散式联邦互联学习

    DFML: Decentralized Federated Mutual Learning

    [https://arxiv.org/abs/2402.01863](https://arxiv.org/abs/2402.01863)

    DFML是一个无服务器的分散式联邦互联学习框架，能够有效地处理模型和数据的异质性，并通过相互学习在客户端之间传授知识，以获得更快的收敛速度和更高的全局准确性。

    

    在现实设备领域中，联邦学习（FL）中的集中式服务器存在通信瓶颈和容易受到单点故障的挑战。此外，现有设备固有地表现出模型和数据的异质性。现有工作缺乏一个能够适应此异质性且不施加架构限制或假定公共数据可用的分散式FL（DFL）框架。为了解决这些问题，我们提出了一个分散式联邦互联学习（DFML）框架，该框架是无服务器的，支持非限制性的异构模型，并避免依赖公共数据。DFML通过相互学习在客户端之间传授知识，并循环改变监督和提取信号的数量来有效处理模型和数据的异质性。广泛的实验结果表明，DFML在收敛速度和全局准确性方面具有一致的有效性，优于普遍存在的方法。

    In the realm of real-world devices, centralized servers in Federated Learning (FL) present challenges including communication bottlenecks and susceptibility to a single point of failure. Additionally, contemporary devices inherently exhibit model and data heterogeneity. Existing work lacks a Decentralized FL (DFL) framework capable of accommodating such heterogeneity without imposing architectural restrictions or assuming the availability of public data. To address these issues, we propose a Decentralized Federated Mutual Learning (DFML) framework that is serverless, supports nonrestrictive heterogeneous models, and avoids reliance on public data. DFML effectively handles model and data heterogeneity through mutual learning, which distills knowledge between clients, and cyclically varying the amount of supervision and distillation signals. Extensive experimental results demonstrate consistent effectiveness of DFML in both convergence speed and global accuracy, outperforming prevalent b
    

