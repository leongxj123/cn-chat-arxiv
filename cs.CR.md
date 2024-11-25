# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Spectrum Breathing: Protecting Over-the-Air Federated Learning Against Interference.](http://arxiv.org/abs/2305.05933) | Spectrum Breathing是一种保护空中联合学习免受干扰的实际方法，通过将随机梯度剪枝和扩频级联起来，以压制干扰而无需扩展带宽。代价是增加的学习延迟。 |

# 详细

[^1]: Spectrum Breathing：保护空中联合学习免受干扰

    Spectrum Breathing: Protecting Over-the-Air Federated Learning Against Interference. (arXiv:2305.05933v1 [cs.LG])

    [http://arxiv.org/abs/2305.05933](http://arxiv.org/abs/2305.05933)

    Spectrum Breathing是一种保护空中联合学习免受干扰的实际方法，通过将随机梯度剪枝和扩频级联起来，以压制干扰而无需扩展带宽。代价是增加的学习延迟。

    

    联合学习是一种从分布式移动数据中蒸馏人工智能的广泛应用范例。但联合学习在移动网络中的部署可能会受到邻近单元或干扰源的干扰而受损。现有的干扰抑制技术需要多单元合作或至少需要昂贵的干扰通道状态信息。另一方面，将干扰视为噪声进行功率控制可能并不有效，由于预算限制，也由于这种机制可能会触发干扰源的反制措施。作为保护空中联合学习免受干扰的实际方法，我们提出了Spectrum Breathing，它将随机梯度剪枝和扩频级联起来，以压制干扰而无需扩展带宽。代价是通过利用剪枝导致学习速度优雅降低而增加的学习延迟。我们将两个操作同步，以保证它们的级别是相互对应的。

    Federated Learning (FL) is a widely embraced paradigm for distilling artificial intelligence from distributed mobile data. However, the deployment of FL in mobile networks can be compromised by exposure to interference from neighboring cells or jammers. Existing interference mitigation techniques require multi-cell cooperation or at least interference channel state information, which is expensive in practice. On the other hand, power control that treats interference as noise may not be effective due to limited power budgets, and also that this mechanism can trigger countermeasures by interference sources. As a practical approach for protecting FL against interference, we propose Spectrum Breathing, which cascades stochastic-gradient pruning and spread spectrum to suppress interference without bandwidth expansion. The cost is higher learning latency by exploiting the graceful degradation of learning speed due to pruning. We synchronize the two operations such that their levels are contr
    

