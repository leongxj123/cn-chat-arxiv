# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TACOS: Topology-Aware Collective Algorithm Synthesizer for Distributed Training.](http://arxiv.org/abs/2304.05301) | TACOS 是一个能够自动合成任意输入网络拓扑的面向拓扑结构的集合合成器。与基准算法相比，TACOS 合成的 All-Reduce 算法速度提高了 3.73 倍，为 512-NPU 系统合成集体算法只需 6.1 分钟。 |

# 详细

[^1]: TACOS: 面向拓扑结构的分布式训练集合算法合成器

    TACOS: Topology-Aware Collective Algorithm Synthesizer for Distributed Training. (arXiv:2304.05301v1 [cs.DC])

    [http://arxiv.org/abs/2304.05301](http://arxiv.org/abs/2304.05301)

    TACOS 是一个能够自动合成任意输入网络拓扑的面向拓扑结构的集合合成器。与基准算法相比，TACOS 合成的 All-Reduce 算法速度提高了 3.73 倍，为 512-NPU 系统合成集体算法只需 6.1 分钟。

    

    集群之间的集合通讯是分布式训练不可或缺的一部分。运行面向拓扑结构的集合算法对于优化通讯性能以最小化拥塞是至关重要的。目前，此类算法仅适用于一小部分简单拓扑结构，限制了训练集群中采用的拓扑结构并处理由于网络故障而产生的不规则拓扑结构。 本文提出了 TACOS，这是一个可自动合成任意输入网络拓扑的面向拓扑结构的集合合成器。TACOS 合成的 All-Reduce 算法比基线算法快了 3.73 倍，并为 512-NPU 系统合成集体算法仅需 6.1 分钟。

    Collective communications are an indispensable part of distributed training. Running a topology-aware collective algorithm is crucial for optimizing communication performance by minimizing congestion. Today such algorithms only exist for a small set of simple topologies, limiting the topologies employed in training clusters and handling irregular topologies due to network failures. In this paper, we propose TACOS, an automated topology-aware collective synthesizer for arbitrary input network topologies. TACOS synthesized 3.73x faster All-Reduce algorithm over baselines, and synthesized collective algorithms for 512-NPU system in just 6.1 minutes.
    

