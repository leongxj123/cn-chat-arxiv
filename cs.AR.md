# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NeuraLUT: Hiding Neural Network Density in Boolean Synthesizable Functions](https://arxiv.org/abs/2403.00849) | 改进了FPGA加速神经网络推断任务的方法，提出将整个子网络映射到单个LUT中，使得神经网络拓扑和精度不再影响生成的查找表的大小。 |

# 详细

[^1]: NeuraLUT: 在Boolean合成函数中隐藏神经网络密度

    NeuraLUT: Hiding Neural Network Density in Boolean Synthesizable Functions

    [https://arxiv.org/abs/2403.00849](https://arxiv.org/abs/2403.00849)

    改进了FPGA加速神经网络推断任务的方法，提出将整个子网络映射到单个LUT中，使得神经网络拓扑和精度不再影响生成的查找表的大小。

    

    可编程门阵列（FPGA）加速器已经证明在处理延迟和资源关键的深度神经网络（DNN）推断任务方面取得了成功。神经网络中计算密集度最高的操作之一是特征和权重向量之间的点积。因此，一些先前的FPGA加速工作提出将具有量化输入和输出的神经元直接映射到查找表（LUTs）以进行硬件实现。在这些工作中，神经元的边界与LUTs的边界重合。我们建议放宽这些边界，将整个子网络映射到单个LUT。由于子网络被吸收到LUT中，分区内的神经网络拓扑和精度不会影响生成的查找表的大小。因此，我们在每个分区内使用具有浮点精度的全连接层，这些层受益于成为通用函数逼近器。

    arXiv:2403.00849v1 Announce Type: cross  Abstract: Field-Programmable Gate Array (FPGA) accelerators have proven successful in handling latency- and resource-critical deep neural network (DNN) inference tasks. Among the most computationally intensive operations in a neural network (NN) is the dot product between the feature and weight vectors. Thus, some previous FPGA acceleration works have proposed mapping neurons with quantized inputs and outputs directly to lookup tables (LUTs) for hardware implementation. In these works, the boundaries of the neurons coincide with the boundaries of the LUTs. We propose relaxing these boundaries and mapping entire sub-networks to a single LUT. As the sub-networks are absorbed within the LUT, the NN topology and precision within a partition do not affect the size of the lookup tables generated. Therefore, we utilize fully connected layers with floating-point precision inside each partition, which benefit from being universal function approximators, 
    

