# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Automated Inference of Graph Transformation Rules](https://arxiv.org/abs/2404.02692) | 提出了一种新颖的图形转换模型构建方法，结合生成和动态观点，实现完全自动化的数据驱动模型推理，通过压缩一组转换成一组规则，允许模型展示超出输入范围的行为。 |
| [^2] | [MLTCP: Congestion Control for DNN Training](https://arxiv.org/abs/2402.09589) | MLTCP是一种用于加速共享GPU集群中的DNN训练作业的拥塞控制技术，通过在每个训练迭代发送的字节数进行缩放，使不同作业的流能够高效利用网络极大地加快训练作业的完成时间。 |

# 详细

[^1]: 图形转换规则的自动推理

    Automated Inference of Graph Transformation Rules

    [https://arxiv.org/abs/2404.02692](https://arxiv.org/abs/2404.02692)

    提出了一种新颖的图形转换模型构建方法，结合生成和动态观点，实现完全自动化的数据驱动模型推理，通过压缩一组转换成一组规则，允许模型展示超出输入范围的行为。

    

    在生命科学领域可用数据的爆炸性增长推动了对富有表现力模型和计算方法日益增长的需求。图形转换是一种具有广泛应用的动态系统模型。我们引入了一种新颖的图形转换模型构建方法，将生成和动态观点结合起来，以提供一个完全自动化的数据驱动模型推理方法。该方法接受作为动态属性的输入，给定为由显式转换编码的动态的“快照”，并构建一个兼容的模型。获得的模型被保证是最小的，因此将该方法规范为模型压缩（将一组转换压缩为一组规则）的方法。压缩对有损情况很宽容，即允许构建的模型展示超出输入转换范围的行为，从而建议完成输入动态的方法。

    arXiv:2404.02692v1 Announce Type: cross  Abstract: The explosion of data available in life sciences is fueling an increasing demand for expressive models and computational methods. Graph transformation is a model for dynamic systems with a large variety of applications. We introduce a novel method of the graph transformation model construction, combining generative and dynamical viewpoints to give a fully automated data-driven model inference method.   The method takes the input dynamical properties, given as a "snapshot" of the dynamics encoded by explicit transitions, and constructs a compatible model. The obtained model is guaranteed to be minimal, thus framing the approach as model compression (from a set of transitions into a set of rules). The compression is permissive to a lossy case, where the constructed model is allowed to exhibit behavior outside of the input transitions, thus suggesting a completion of the input dynamics.   The task of graph transformation model inference i
    
[^2]: MLTCP:用于DNN训练的拥塞控制技术

    MLTCP: Congestion Control for DNN Training

    [https://arxiv.org/abs/2402.09589](https://arxiv.org/abs/2402.09589)

    MLTCP是一种用于加速共享GPU集群中的DNN训练作业的拥塞控制技术，通过在每个训练迭代发送的字节数进行缩放，使不同作业的流能够高效利用网络极大地加快训练作业的完成时间。

    

    我们提出了MLTCP，一种技术来增强当前的拥塞控制算法，以加速在共享GPU集群中进行的DNN训练作业。MLTCP使竞争网络带宽的作业的通信阶段相互交错，从而高效利用网络。MLTCP的核心是一个基于关键概念洞察的非常简单的原则：DNN训练流应该根据每个训练迭代发送的字节数来缩放其拥塞窗口大小。我们展示了将这个原则整合到当前的拥塞控制协议中是直接的：通过在Reno、CUBIC或DCQCN中添加30-60行代码，MLTCP可以在几个训练迭代内将不同作业的流稳定地转化为交错状态，无论竞争流的数量或每个流的开始时间如何。我们对流行的DNN训练作业进行的实验表明，启用MLTCP可以加快平均和99th pe的结束时间

    arXiv:2402.09589v1 Announce Type: cross  Abstract: We present MLTCP, a technique to augment today's congestion control algorithms to accelerate DNN training jobs in shared GPU clusters. MLTCP enables the communication phases of jobs that compete for network bandwidth to interleave with each other, thereby utilizing the network efficiently. At the heart of MLTCP lies a very simple principle based on a key conceptual insight: DNN training flows should scale their congestion window size based on the number of bytes sent at each training iteration. We show that integrating this principle into today's congestion control protocols is straightforward: by adding 30-60 lines of code to Reno, CUBIC, or DCQCN, MLTCP stabilizes flows of different jobs into an interleaved state within a few training iterations, regardless of the number of competing flows or the start time of each flow. Our experiments with popular DNN training jobs demonstrate that enabling MLTCP accelerates the average and 99th pe
    

