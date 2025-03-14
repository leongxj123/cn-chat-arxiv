# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MoE-Infinity: Activation-Aware Expert Offloading for Efficient MoE Serving.](http://arxiv.org/abs/2401.14361) | MoE-Infinity是一种成本高效的MoE服务系统，通过激活感知的专家卸载和缓存技术，显著降低了延迟，并提高了成本性能。 |

# 详细

[^1]: MoE-Infinity：用于高效MoE服务的激活感知专家卸载系统

    MoE-Infinity: Activation-Aware Expert Offloading for Efficient MoE Serving. (arXiv:2401.14361v1 [cs.LG])

    [http://arxiv.org/abs/2401.14361](http://arxiv.org/abs/2401.14361)

    MoE-Infinity是一种成本高效的MoE服务系统，通过激活感知的专家卸载和缓存技术，显著降低了延迟，并提高了成本性能。

    

    本文介绍了MoE-Infinity，一种成本高效的专家混合(MoE)服务系统，实现了激活感知的专家卸载。MoE-Infinity具有序列级专家激活追踪的特点，这是一种擅长识别稀疏激活并捕捉MoE推理的时间局部性的新方法。通过分析这些追踪，MoE-Infinity执行了新颖的激活感知专家预取和缓存，大大降低了通常与卸载专家相关的延迟开销，提高了成本性能。在一个集群中进行的大量实验表明，MoE-Infinity优于许多现有的系统和方法，对于各种MoEs，将延迟降低了420倍，将部署成本降低了8倍以上。MoE-Infinity的源代码可在https://github.com/TorchMoE/MoE-Infinity公开获取。

    This paper presents MoE-Infinity, a cost-efficient mixture-of-expert (MoE) serving system that realizes activation-aware expert offloading. MoE-Infinity features sequence-level expert activation tracing, a new approach adept at identifying sparse activations and capturing the temporal locality of MoE inference. By analyzing these traces, MoE-Infinity performs novel activation-aware expert prefetching and caching, substantially reducing the latency overheads usually associated with offloading experts for improved cost performance. Extensive experiments in a cluster show that MoE-Infinity outperforms numerous existing systems and approaches, reducing latency by 4 20X and decreasing deployment costs by over 8X for various MoEs. MoE-Infinity's source code is publicly available at https://github.com/TorchMoE/MoE-Infinity
    

