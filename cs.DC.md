# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Convergence Analysis of Split Federated Learning on Heterogeneous Data](https://arxiv.org/abs/2402.15166) | 本文填补了分裂联邦学习在各异数据上收敛分析的空白，提供了针对强凸和一般凸目标的SFL收敛分析，收敛速率分别为$O(1/T)$和$O(1/\sqrt[3]{T})。 |
| [^2] | [Hybrid Retrieval-Augmented Generation for Real-time Composition Assistance.](http://arxiv.org/abs/2308.04215) | 提出了一种混合检索增强生成的框架，通过将云模型的检索增强内存整合到客户端模型中，实现实时响应的作曲辅助。 |

# 详细

[^1]: 分布式异构数据上的分裂联邦学习的收敛分析

    Convergence Analysis of Split Federated Learning on Heterogeneous Data

    [https://arxiv.org/abs/2402.15166](https://arxiv.org/abs/2402.15166)

    本文填补了分裂联邦学习在各异数据上收敛分析的空白，提供了针对强凸和一般凸目标的SFL收敛分析，收敛速率分别为$O(1/T)$和$O(1/\sqrt[3]{T})。

    

    分裂联邦学习（SFL）是一种最近的分布式方法，用于在多个客户端之间进行协作模型训练。在SFL中，全局模型通常被分为两部分，其中客户端以并行联邦方式训练一部分，主服务器训练另一部分。尽管最近关于SFL算法发展的研究很多，但SFL的收敛分析在文献中还未有提及，本文旨在弥补这一空白。对SFL进行分析可能比对联邦学习（FL）的分析更具挑战性，这是由于客户端和主服务器之间可能存在双速更新。我们提供了针对异构数据上强凸和一般凸目标的SFL收敛分析。收敛速率分别为$O(1/T)$和$O(1/\sqrt[3]{T})$，其中$T$表示SFL训练的总轮数。我们进一步将分析扩展到非凸目标和一些客户端可能在训练过程中不可用的情况。

    arXiv:2402.15166v1 Announce Type: cross  Abstract: Split federated learning (SFL) is a recent distributed approach for collaborative model training among multiple clients. In SFL, a global model is typically split into two parts, where clients train one part in a parallel federated manner, and a main server trains the other. Despite the recent research on SFL algorithm development, the convergence analysis of SFL is missing in the literature, and this paper aims to fill this gap. The analysis of SFL can be more challenging than that of federated learning (FL), due to the potential dual-paced updates at the clients and the main server. We provide convergence analysis of SFL for strongly convex and general convex objectives on heterogeneous data. The convergence rates are $O(1/T)$ and $O(1/\sqrt[3]{T})$, respectively, where $T$ denotes the total number of rounds for SFL training. We further extend the analysis to non-convex objectives and where some clients may be unavailable during trai
    
[^2]: 实时作曲辅助的混合检索增强生成

    Hybrid Retrieval-Augmented Generation for Real-time Composition Assistance. (arXiv:2308.04215v1 [cs.CL])

    [http://arxiv.org/abs/2308.04215](http://arxiv.org/abs/2308.04215)

    提出了一种混合检索增强生成的框架，通过将云模型的检索增强内存整合到客户端模型中，实现实时响应的作曲辅助。

    

    检索增强模型在提升传统语言模型的上下文理解、整合私人数据和减少幻觉方面显示出了潜力。然而，应用于需要实时响应的任务（如作曲辅助）时，检索增强的大型语言模型所需的处理时间存在挑战。为了克服这一限制，我们提出了Hybrid Retrieval-Augmented Generation (HybridRAG)框架，利用了将客户端模型和云模型结合起来的混合设置。HybridRAG通过异步生成的检索增强内存，将大型语言模型（LLM）在云端生成的检索增强内存整合到客户端模型中。通过整合这种检索增强内存，客户端模型能够生成高效的响应，从LLM的能力中受益。此外，通过异步内存集成，客户端模型能够实时响应用户请求，无需等待云端处理。

    Retrieval augmented models show promise in enhancing traditional language models by improving their contextual understanding, integrating private data, and reducing hallucination. However, the processing time required for retrieval augmented large language models poses a challenge when applying them to tasks that require real-time responses, such as composition assistance.  To overcome this limitation, we propose the Hybrid Retrieval-Augmented Generation (HybridRAG) framework that leverages a hybrid setting that combines both client and cloud models. HybridRAG incorporates retrieval-augmented memory generated asynchronously by a Large Language Model (LLM) in the cloud. By integrating this retrieval augmented memory, the client model acquires the capability to generate highly effective responses, benefiting from the LLM's capabilities. Furthermore, through asynchronous memory integration, the client model is capable of delivering real-time responses to user requests without the need to 
    

