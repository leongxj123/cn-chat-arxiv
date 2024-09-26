# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Collaboration in Distributed Parameter Estimation with Resource Constraints.](http://arxiv.org/abs/2307.06442) | 在资源约束下的分布参数估计中，我们研究了传感器/代理数据收集和协作策略，通过最大化费舍尔信息或最小化Cramer-Rao界来解决传感器/代理的数据收集和协作策略设计问题。 |
| [^2] | [Fast Distributed Inference Serving for Large Language Models.](http://arxiv.org/abs/2305.05920) | FastServe是一种针对大型语言模型的分布式推理服务系统，利用抢占式调度和跳过-连接多级反馈队列，最小化模型推断的作业完成时间(JCT)。 |

# 详细

[^1]: 在资源约束下的分布参数估计中的协作研究

    On Collaboration in Distributed Parameter Estimation with Resource Constraints. (arXiv:2307.06442v1 [cs.LG])

    [http://arxiv.org/abs/2307.06442](http://arxiv.org/abs/2307.06442)

    在资源约束下的分布参数估计中，我们研究了传感器/代理数据收集和协作策略，通过最大化费舍尔信息或最小化Cramer-Rao界来解决传感器/代理的数据收集和协作策略设计问题。

    

    我们研究了考虑资源约束和不同传感器/代理收集的观测之间的相关性的参数估计的传感器/代理数据收集和协作策略。具体地，我们考虑了一组传感器/代理，每个传感器/代理样本来自多元高斯分布的不同变量，并且具有不同的估计目标，我们将传感器/代理的数据收集和协作策略设计问题阐述为费舍尔信息最大化（或Cramer-Rao界最小化）问题。当变量之间的相关性知识可用时，我们可以分析地识别出两个特定情况：（1）不能利用样本之间的相关性知识进行协作估计的情况，（2）最优数据收集策略涉及投资有限资源以协作采样和转移已知统计信息的情况。

    We study sensor/agent data collection and collaboration policies for parameter estimation, accounting for resource constraints and correlation between observations collected by distinct sensors/agents. Specifically, we consider a group of sensors/agents each samples from different variables of a multivariate Gaussian distribution and has different estimation objectives, and we formulate a sensor/agent's data collection and collaboration policy design problem as a Fisher information maximization (or Cramer-Rao bound minimization) problem. When the knowledge of correlation between variables is available, we analytically identify two particular scenarios: (1) where the knowledge of the correlation between samples cannot be leveraged for collaborative estimation purposes and (2) where the optimal data collection policy involves investing scarce resources to collaboratively sample and transfer information that is not of immediate interest and whose statistics are already known, with the sol
    
[^2]: 大型语言模型快速分布式推断服务

    Fast Distributed Inference Serving for Large Language Models. (arXiv:2305.05920v1 [cs.LG])

    [http://arxiv.org/abs/2305.05920](http://arxiv.org/abs/2305.05920)

    FastServe是一种针对大型语言模型的分布式推理服务系统，利用抢占式调度和跳过-连接多级反馈队列，最小化模型推断的作业完成时间(JCT)。

    

    大型语言模型(LLM)推动了以ChatGPT为代表的新一代互动AI应用程序的发展。这些应用程序的交互性要求模型推断的低作业完成时间(JCT)。现有的LLM服务系统使用的是运行到完成的处理方式，存在头部阻塞和长JCT的问题。我们提出了FastServe，一种针对LLMs的分布式推理服务系统。FastServe利用LLM推理的自回归模式，以每个输出标记的粒度实现抢占式，使用新颖的跳过-连接多级反馈队列调度器最小化JCT。基于LLM推理的新半信息不可知设置，调度程序利用输入长度信息来为每个到达作业分配适当的初始队列来连接。高于所连接队列的优先级队列被跳过以减少降级。我们设计了一种高效的GPU内存管理机制，以提前清除不再使用的GPU缓存，并对常用模型进行缓存。

    Large language models (LLMs) power a new generation of interactive AI applications exemplified by ChatGPT. The interactive nature of these applications demand low job completion time (JCT) for model inference. Existing LLM serving systems use run-to-completion processing for inference jobs, which suffers from head-of-line blocking and long JCT. We present FastServe, a distributed inference serving system for LLMs. FastServe exploits the autoregressive pattern of LLM inference to enable preemption at the granularity of each output token. FastServe uses preemptive scheduling to minimize JCT with a novel skip-join Multi-Level Feedback Queue scheduler. Based on the new semi information-agnostic setting of LLM inference, the scheduler leverages the input length information to assign an appropriate initial queue for each arrival job to join. The higher priority queues than the joined queue are skipped to reduce demotions. We design an efficient GPU memory management mechanism that proactivel
    

