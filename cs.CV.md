# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Simplified Diffusion Schr\"odinger Bridge](https://arxiv.org/abs/2403.14623) | 介绍了简化后的扩散薛定谔桥（DSB），通过与基于得分的生成模型（SGM）的统一解决了复杂数据生成中的限制，提高了性能并加快了收敛速度。 |
| [^2] | [EdgeOL: Efficient in-situ Online Learning on Edge Devices.](http://arxiv.org/abs/2401.16694) | 本文提出了EdgeOL，一种边缘在线学习框架，通过内部和外部调优来优化推理准确性、微调执行时间和能量效率，在边缘设备上实现了显著的性能提升。 |

# 详细

[^1]: 简化扩散薛定谔桥

    Simplified Diffusion Schr\"odinger Bridge

    [https://arxiv.org/abs/2403.14623](https://arxiv.org/abs/2403.14623)

    介绍了简化后的扩散薛定谔桥（DSB），通过与基于得分的生成模型（SGM）的统一解决了复杂数据生成中的限制，提高了性能并加快了收敛速度。

    

    这篇论文介绍了一种新的理论简化扩散薛定谔桥（DSB），便于将其与基于得分的生成模型（SGM）统一起来，解决了DSB在复杂数据生成方面的局限性，实现更快的收敛速度和增强的性能。通过将SGM作为DSB的初始解决方案，我们的方法利用了这两个框架的优势，确保了更高效的训练过程，并改进了SGM的性能。我们还提出了一种重新参数化技术，尽管存在理论近似，但实际上提高了网络的拟合能力。我们进行了大量的实验证实，证实了简化的DSB的有效性，展示了其显著的改进。我们相信这项工作的贡献为先进的生成建模铺平了道路。

    arXiv:2403.14623v1 Announce Type: new  Abstract: This paper introduces a novel theoretical simplification of the Diffusion Schr\"odinger Bridge (DSB) that facilitates its unification with Score-based Generative Models (SGMs), addressing the limitations of DSB in complex data generation and enabling faster convergence and enhanced performance. By employing SGMs as an initial solution for DSB, our approach capitalizes on the strengths of both frameworks, ensuring a more efficient training process and improving the performance of SGM. We also propose a reparameterization technique that, despite theoretical approximations, practically improves the network's fitting capabilities. Our extensive experimental evaluations confirm the effectiveness of the simplified DSB, demonstrating its significant improvements. We believe the contributions of this work pave the way for advanced generative modeling. The code is available at https://github.com/tzco/Simplified-Diffusion-Schrodinger-Bridge.
    
[^2]: EdgeOL: 边缘设备上高效的原位在线学习

    EdgeOL: Efficient in-situ Online Learning on Edge Devices. (arXiv:2401.16694v1 [cs.LG])

    [http://arxiv.org/abs/2401.16694](http://arxiv.org/abs/2401.16694)

    本文提出了EdgeOL，一种边缘在线学习框架，通过内部和外部调优来优化推理准确性、微调执行时间和能量效率，在边缘设备上实现了显著的性能提升。

    

    新兴应用，如机器人辅助养老和物体识别，通常采用深度学习神经网络模型，并且自然需要：i) 处理实时推理请求和ii) 适应可能的部署场景变化。在线模型微调被广泛采用以满足这些需求。然而，微调会导致显著的能量消耗，使其难以部署在边缘设备上。在本文中，我们提出了EdgeOL，一种边缘在线学习框架，通过内部和外部调优来优化推理准确性、微调执行时间和能量效率。实验结果显示，EdgeOL平均减少了82%的微调执行时间，74%的能量消耗，并提高了平均推理准确率1.70%，相对于即时在线学习策略。

    Emerging applications, such as robot-assisted eldercare and object recognition, generally employ deep learning neural networks (DNNs) models and naturally require: i) handling streaming-in inference requests and ii) adapting to possible deployment scenario changes. Online model fine-tuning is widely adopted to satisfy these needs. However, fine-tuning involves significant energy consumption, making it challenging to deploy on edge devices. In this paper, we propose EdgeOL, an edge online learning framework that optimizes inference accuracy, fine-tuning execution time, and energy efficiency through both inter-tuning and intra-tuning optimizations. Experimental results show that, on average, EdgeOL reduces overall fine-tuning execution time by 82%, energy consumption by 74%, and improves average inference accuracy by 1.70% over the immediate online learning strategy.
    

