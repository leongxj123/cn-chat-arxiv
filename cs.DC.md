# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [EdgeOL: Efficient in-situ Online Learning on Edge Devices.](http://arxiv.org/abs/2401.16694) | 本文提出了EdgeOL，一种边缘在线学习框架，通过内部和外部调优来优化推理准确性、微调执行时间和能量效率，在边缘设备上实现了显著的性能提升。 |

# 详细

[^1]: EdgeOL: 边缘设备上高效的原位在线学习

    EdgeOL: Efficient in-situ Online Learning on Edge Devices. (arXiv:2401.16694v1 [cs.LG])

    [http://arxiv.org/abs/2401.16694](http://arxiv.org/abs/2401.16694)

    本文提出了EdgeOL，一种边缘在线学习框架，通过内部和外部调优来优化推理准确性、微调执行时间和能量效率，在边缘设备上实现了显著的性能提升。

    

    新兴应用，如机器人辅助养老和物体识别，通常采用深度学习神经网络模型，并且自然需要：i) 处理实时推理请求和ii) 适应可能的部署场景变化。在线模型微调被广泛采用以满足这些需求。然而，微调会导致显著的能量消耗，使其难以部署在边缘设备上。在本文中，我们提出了EdgeOL，一种边缘在线学习框架，通过内部和外部调优来优化推理准确性、微调执行时间和能量效率。实验结果显示，EdgeOL平均减少了82%的微调执行时间，74%的能量消耗，并提高了平均推理准确率1.70%，相对于即时在线学习策略。

    Emerging applications, such as robot-assisted eldercare and object recognition, generally employ deep learning neural networks (DNNs) models and naturally require: i) handling streaming-in inference requests and ii) adapting to possible deployment scenario changes. Online model fine-tuning is widely adopted to satisfy these needs. However, fine-tuning involves significant energy consumption, making it challenging to deploy on edge devices. In this paper, we propose EdgeOL, an edge online learning framework that optimizes inference accuracy, fine-tuning execution time, and energy efficiency through both inter-tuning and intra-tuning optimizations. Experimental results show that, on average, EdgeOL reduces overall fine-tuning execution time by 82%, energy consumption by 74%, and improves average inference accuracy by 1.70% over the immediate online learning strategy.
    

