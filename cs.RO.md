# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tactile-based Object Retrieval From Granular Media](https://arxiv.org/abs/2402.04536) | 这项研究介绍了一种基于触觉反馈的机器人操作方法，用于在颗粒介质中检索埋藏的物体。通过模拟传感器噪声进行端到端训练，实现了自然出现的学习推动行为，并成功将其迁移到实际硬件上。 |

# 详细

[^1]: 基于触觉的从颗粒介质中检索物体的研究

    Tactile-based Object Retrieval From Granular Media

    [https://arxiv.org/abs/2402.04536](https://arxiv.org/abs/2402.04536)

    这项研究介绍了一种基于触觉反馈的机器人操作方法，用于在颗粒介质中检索埋藏的物体。通过模拟传感器噪声进行端到端训练，实现了自然出现的学习推动行为，并成功将其迁移到实际硬件上。

    

    我们介绍了一种名为GEOTACT的机器人操作方法，能够在颗粒介质中检索埋藏的物体。这是一项具有挑战性的任务，因为需要与颗粒介质进行交互，并且仅依靠触觉反馈来完成，因为一个埋藏的物体可能完全被视觉隐藏。在这种环境中，触觉反馈本身具有挑战性，因为需要与周围介质进行普遍接触，并且由触觉读数引起的固有噪声水平。为了解决这些挑战，我们使用了一种通过模拟传感器噪声进行端到端训练的学习方法。我们展示了我们的问题表述导致了学习推动行为的自然出现，操作器使用这些行为来减少不确定性并将物体引导到稳定的抓取位置，尽管存在假的和噪声的触觉读数。我们还引入了一种培训方案，可以在仿真中学习这些行为，并在实际硬件上进行零样本迁移。据我们所知，GEOTACT是第一个这样的方法。

    We introduce GEOTACT, a robotic manipulation method capable of retrieving objects buried in granular media. This is a challenging task due to the need to interact with granular media, and doing so based exclusively on tactile feedback, since a buried object can be completely hidden from vision. Tactile feedback is in itself challenging in this context, due to ubiquitous contact with the surrounding media, and the inherent noise level induced by the tactile readings. To address these challenges, we use a learning method trained end-to-end with simulated sensor noise. We show that our problem formulation leads to the natural emergence of learned pushing behaviors that the manipulator uses to reduce uncertainty and funnel the object to a stable grasp despite spurious and noisy tactile readings. We also introduce a training curriculum that enables learning these behaviors in simulation, followed by zero-shot transfer to real hardware. To the best of our knowledge, GEOTACT is the first meth
    

