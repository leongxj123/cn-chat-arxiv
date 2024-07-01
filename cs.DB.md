# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DLRover: An Elastic Deep Training Extension with Auto Job Resource Recommendation.](http://arxiv.org/abs/2304.01468) | DLRover是一个自动配置初始资源并实时调整资源的分布式深度学习框架，解决了资源共享和手动配置带来的问题。 |

# 详细

[^1]: DLRover：一种具有自动作业资源推荐的弹性深度训练扩展

    DLRover: An Elastic Deep Training Extension with Auto Job Resource Recommendation. (arXiv:2304.01468v1 [cs.DC])

    [http://arxiv.org/abs/2304.01468](http://arxiv.org/abs/2304.01468)

    DLRover是一个自动配置初始资源并实时调整资源的分布式深度学习框架，解决了资源共享和手动配置带来的问题。

    

    由于在云平台上进行资源共享可以提高资源利用率并降低总体成本，因此云仍然是分布式深度学习（DL）训练作业的流行平台。然而，此类共享也为DL训练作业带来了多重挑战，例如高优先级作业可能会影响、甚至中断低优先级作业。同时，大多数现有的分布式DL训练系统要求用户在作业提交之前手动配置作业的资源（即分配给每个节点的节点数和CPU、内存等资源），并且不能在运行时调整作业的资源。作业的资源配置会深刻影响该作业的性能（例如训练吞吐量、资源利用率和完成率）。然而，这通常会导致作业性能不佳，因为用户在大多数情况下无法提供最佳的资源配置。DLRover是一种分布式DL框架，可以自动配置DL作业的初始资源并动态调整作业的资源。

    The cloud is still a popular platform for distributed deep learning (DL) training jobs since resource sharing in the cloud can improve resource utilization and reduce overall costs. However, such sharing also brings multiple challenges for DL training jobs, e.g., high-priority jobs could impact, even interrupt, low-priority jobs. Meanwhile, most existing distributed DL training systems require users to configure the resources (i.e., the number of nodes and resources like CPU and memory allocated to each node) of jobs manually before job submission and can not adjust the job's resources during the runtime. The resource configuration of a job deeply affect this job's performance (e.g., training throughput, resource utilization, and completion rate). However, this usually leads to poor performance of jobs since users fail to provide optimal resource configuration in most cases. \system~is a distributed DL framework can auto-configure a DL job's initial resources and dynamically tune the j
    

