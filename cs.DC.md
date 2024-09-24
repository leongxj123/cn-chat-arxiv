# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ForestColl: Efficient Collective Communications on Heterogeneous Network Fabrics](https://arxiv.org/abs/2402.06787) | ForestColl是一种针对任意网络拓扑生成高效调度的工具，通过构建广播/聚合生成跨越树的通信调度，实现了理论上的最小网络拥塞，并在实验中表现出高于供应商自带通信库的性能。 |
| [^2] | [Fast Distributed Inference Serving for Large Language Models.](http://arxiv.org/abs/2305.05920) | FastServe是一种针对大型语言模型的分布式推理服务系统，利用抢占式调度和跳过-连接多级反馈队列，最小化模型推断的作业完成时间(JCT)。 |

# 详细

[^1]: ForestColl: 异构网络结构上高效的集合通信

    ForestColl: Efficient Collective Communications on Heterogeneous Network Fabrics

    [https://arxiv.org/abs/2402.06787](https://arxiv.org/abs/2402.06787)

    ForestColl是一种针对任意网络拓扑生成高效调度的工具，通过构建广播/聚合生成跨越树的通信调度，实现了理论上的最小网络拥塞，并在实验中表现出高于供应商自带通信库的性能。

    

    随着现代深度神经网络模型越来越大，加速器之间的集合通信（如allreduce等）成为一个重要的性能瓶颈。在当今高度多样化和异构的网络结构下设计高效的通信调度是一项具有挑战性的任务。本文提出了一种名为ForestColl的工具，它能够为任意网络拓扑生成高效的调度。ForestColl使用广播/聚合生成跨越树作为通信调度，实现了理论上的最小网络拥塞。其调度生成运行在强多项式时间内，且具有高扩展性。ForestColl支持包括交换网络和直接连接在内的任何网络结构，以及任何网络图结构。我们在多集群的AMD MI250和NVIDIA A100平台上评估了ForestColl。与供应商自己优化的通信库RCCL和NCCL相比，ForestColl的调度性能提高了高达52％。ForestColl还优于其他...

    As modern DNN models grow ever larger, collective communications between the accelerators (allreduce, etc.) emerge as a significant performance bottleneck. Designing efficient communication schedules is challenging given today's highly diverse and heterogeneous network fabrics. In this paper, we present ForestColl, a tool that generates efficient schedules for any network topology. ForestColl constructs broadcast/aggregation spanning trees as the communication schedule, achieving theoretically minimum network congestion. Its schedule generation runs in strongly polynomial time and is highly scalable. ForestColl supports any network fabrics, including both switching fabrics and direct connections, as well as any network graph structure. We evaluated ForestColl on multi-cluster AMD MI250 and NVIDIA A100 platforms. ForestColl's schedules achieved up to 52\% higher performance compared to the vendors' own optimized communication libraries, RCCL and NCCL. ForestColl also outperforms other s
    
[^2]: 大型语言模型快速分布式推断服务

    Fast Distributed Inference Serving for Large Language Models. (arXiv:2305.05920v1 [cs.LG])

    [http://arxiv.org/abs/2305.05920](http://arxiv.org/abs/2305.05920)

    FastServe是一种针对大型语言模型的分布式推理服务系统，利用抢占式调度和跳过-连接多级反馈队列，最小化模型推断的作业完成时间(JCT)。

    

    大型语言模型(LLM)推动了以ChatGPT为代表的新一代互动AI应用程序的发展。这些应用程序的交互性要求模型推断的低作业完成时间(JCT)。现有的LLM服务系统使用的是运行到完成的处理方式，存在头部阻塞和长JCT的问题。我们提出了FastServe，一种针对LLMs的分布式推理服务系统。FastServe利用LLM推理的自回归模式，以每个输出标记的粒度实现抢占式，使用新颖的跳过-连接多级反馈队列调度器最小化JCT。基于LLM推理的新半信息不可知设置，调度程序利用输入长度信息来为每个到达作业分配适当的初始队列来连接。高于所连接队列的优先级队列被跳过以减少降级。我们设计了一种高效的GPU内存管理机制，以提前清除不再使用的GPU缓存，并对常用模型进行缓存。

    Large language models (LLMs) power a new generation of interactive AI applications exemplified by ChatGPT. The interactive nature of these applications demand low job completion time (JCT) for model inference. Existing LLM serving systems use run-to-completion processing for inference jobs, which suffers from head-of-line blocking and long JCT. We present FastServe, a distributed inference serving system for LLMs. FastServe exploits the autoregressive pattern of LLM inference to enable preemption at the granularity of each output token. FastServe uses preemptive scheduling to minimize JCT with a novel skip-join Multi-Level Feedback Queue scheduler. Based on the new semi information-agnostic setting of LLM inference, the scheduler leverages the input length information to assign an appropriate initial queue for each arrival job to join. The higher priority queues than the joined queue are skipped to reduce demotions. We design an efficient GPU memory management mechanism that proactivel
    

