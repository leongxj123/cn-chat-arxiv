# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Collaborative Inference via Dynamic Composition of Tiny AI Accelerators on MCUs.](http://arxiv.org/abs/2401.08637) | 该论文介绍了Synergy，一个通过动态组合微型AI加速器来进行协作推断的系统，有效地解决了在设备上AI需求不断增长时tinyML面临的关键挑战。Synergy通过提供虚拟计算空间和运行时编排模块，实现了资源的统一虚拟化视图和跨动态/异构加速器的最佳推断，其吞吐量平均提升了8.0倍。 |

# 详细

[^1]: 通过MCU上微型AI加速器的动态组合实现协作推断

    Collaborative Inference via Dynamic Composition of Tiny AI Accelerators on MCUs. (arXiv:2401.08637v1 [cs.DC])

    [http://arxiv.org/abs/2401.08637](http://arxiv.org/abs/2401.08637)

    该论文介绍了Synergy，一个通过动态组合微型AI加速器来进行协作推断的系统，有效地解决了在设备上AI需求不断增长时tinyML面临的关键挑战。Synergy通过提供虚拟计算空间和运行时编排模块，实现了资源的统一虚拟化视图和跨动态/异构加速器的最佳推断，其吞吐量平均提升了8.0倍。

    

    微型AI加速器的出现为深度神经网络在极限边缘上的部署提供了机会，提供了较低的延迟、较低的功耗成本和改进的隐私保护。尽管取得了这些进展，但由于这些加速器的固有限制，如有限的内存和单设备焦点，仍存在挑战。本文介绍了Synergy，一个能够为多租户模型动态组合微型AI加速器的系统，有效解决了对于设备上AI的需求不断增长时tinyML面临的关键挑战。Synergy的一个关键特性是其提供了虚拟计算空间，为资源提供了统一的虚拟化视图，从而实现了对物理设备的高效任务映射。Synergy的运行时编排模块确保了跨动态和异构加速器的最佳推断。我们的评估结果显示，与基准相比，Synergy的吞吐量平均提升了8.0倍。

    The advent of tiny AI accelerators opens opportunities for deep neural network deployment at the extreme edge, offering reduced latency, lower power cost, and improved privacy in on-device ML inference. Despite these advancements, challenges persist due to inherent limitations of these accelerators, such as restricted onboard memory and single-device focus. This paper introduces Synergy, a system that dynamically composes tiny AI accelerators for multi-tenant models, effectively addressing tinyML's critical challenges for the increasing demand for on-device AI. A key feature of Synergy is its virtual computing space, providing a unified, virtualized view of resources and enabling efficient task mapping to physical devices. Synergy's runtime orchestration module ensures optimal inference across dynamic and heterogeneous accelerators. Our evaluations with 7 baselines and 8 models demonstrate that Synergy improves throughput by an average of 8.0X compared to baselines.
    

