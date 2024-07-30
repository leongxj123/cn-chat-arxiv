# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimized Network Architectures for Large Language Model Training with Billions of Parameters.](http://arxiv.org/abs/2307.12169) | 本文提出了一种优化的网络架构，用于训练拥有数十亿参数的大型语言模型。这个架构根据语言模型的通信需求，将集群分割成一组通过非阻塞高带宽互连的GPU集合，并通过轨道连接仅连接具有通信需求的GPU，从而降低网络成本高达75％，同时不影响训练性能。 |

# 详细

[^1]: 用于训练拥有数十亿参数的大型语言模型的优化网络架构

    Optimized Network Architectures for Large Language Model Training with Billions of Parameters. (arXiv:2307.12169v1 [cs.NI])

    [http://arxiv.org/abs/2307.12169](http://arxiv.org/abs/2307.12169)

    本文提出了一种优化的网络架构，用于训练拥有数十亿参数的大型语言模型。这个架构根据语言模型的通信需求，将集群分割成一组通过非阻塞高带宽互连的GPU集合，并通过轨道连接仅连接具有通信需求的GPU，从而降低网络成本高达75％，同时不影响训练性能。

    

    本文挑战了为训练大型语言模型（LLMs）构建任意到任意网络的传统范式。我们展示了LLMs呈现出一种独特的通信模式，在其中，只有小组的GPU需要高带宽的任意到任意通信，以实现接近最优的训练性能。在这些GPU小组之间，通信非常微不足道、稀疏且均匀。我们提出了一个新的网络架构，紧密匹配LLMs的通信需求。我们的架构将集群分割为一组通过非阻塞任意到任意高带宽互连的GPU集合，我们称之为HB域。在HB域之间，网络只连接具有通信需求的GPU。我们将这种网络连接称为“仅轨道连接”，并展示了我们的架构相对于最先进的任意到任意Clos网络可以将网络成本降低高达75％，同时不损害LLM训练的性能。

    This paper challenges the well-established paradigm for building any-to-any networks for training Large Language Models (LLMs). We show that LLMs exhibit a unique communication pattern where only small groups of GPUs require high-bandwidth any-to-any communication within them, to achieve near-optimal training performance. Across these groups of GPUs, the communication is insignificant, sparse, and homogeneous. We propose a new network architecture that closely resembles the communication requirement of LLMs. Our architecture partitions the cluster into sets of GPUs interconnected with non-blocking any-to-any high-bandwidth interconnects that we call HB domains. Across the HB domains, the network only connects GPUs with communication demands. We call this network a "rail-only" connection, and show that our proposed architecture reduces the network cost by up to 75% compared to the state-of-the-art any-to-any Clos networks without compromising the performance of LLM training.
    

