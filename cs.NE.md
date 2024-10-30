# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [S-TLLR: STDP-inspired Temporal Local Learning Rule for Spiking Neural Networks.](http://arxiv.org/abs/2306.15220) | S-TLLR是一个受到STDP机制启发的时间局部学习规则，可以用于训练脉冲神经网络，同时考虑到了因果和非因果关系。 |

# 详细

[^1]: S-TLLR: 受到时间局部学习规则的STDP启发的脉冲神经网络

    S-TLLR: STDP-inspired Temporal Local Learning Rule for Spiking Neural Networks. (arXiv:2306.15220v1 [cs.NE])

    [http://arxiv.org/abs/2306.15220](http://arxiv.org/abs/2306.15220)

    S-TLLR是一个受到STDP机制启发的时间局部学习规则，可以用于训练脉冲神经网络，同时考虑到了因果和非因果关系。

    

    脉冲神经网络（SNN）是可用于边缘智能的生物学合理模型，特别适用于顺序学习任务。然而，SNN的训练面临着精确的时间和空间信用分配的挑战。尽管BPTT算法是解决这些问题最常用的方法，但由于其时间依赖性，它产生了较高的计算成本。此外，BPTT及其近似仅利用从脉冲活动中导出的因果信息来计算突触更新，从而忽略了非因果关系。在这项工作中，我们提出了S-TLLR，这是一种受到Spike-Timing Dependent Plasticity（STDP）机制启发的新型三因素时间局部学习规则，旨在用于事件驱动学习任务的SNN训练。S-TLLR同时考虑了前后突触之间的因果和非因果关系。

    Spiking Neural Networks (SNNs) are biologically plausible models that have been identified as potentially apt for the deployment for energy-efficient intelligence at the edge, particularly for sequential learning tasks. However, training of SNNs poses a significant challenge due to the necessity for precise temporal and spatial credit assignment. Back-propagation through time (BPTT) algorithm, whilst being the most widely used method for addressing these issues, incurs a high computational cost due to its temporal dependency. Moreover, BPTT and its approximations solely utilize causal information derived from the spiking activity to compute the synaptic updates, thus neglecting non-causal relationships. In this work, we propose S-TLLR, a novel three-factor temporal local learning rule inspired by the Spike-Timing Dependent Plasticity (STDP) mechanism, aimed at training SNNs on event-based learning tasks. S-TLLR considers both causal and non-causal relationships between pre and post-syn
    

