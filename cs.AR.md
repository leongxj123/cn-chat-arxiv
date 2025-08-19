# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Negative Feedback Training: A Novel Concept to Improve Robustness of NVCiM DNN Accelerators.](http://arxiv.org/abs/2305.14561) | 本文介绍了一种新的训练方法，使用负反馈机制来增强DNN模型的鲁棒性，特别是在存在设备变异的情况下。 |

# 详细

[^1]: 负反馈训练：提高NVCiM DNN加速器鲁棒性的新概念

    Negative Feedback Training: A Novel Concept to Improve Robustness of NVCiM DNN Accelerators. (arXiv:2305.14561v1 [cs.LG])

    [http://arxiv.org/abs/2305.14561](http://arxiv.org/abs/2305.14561)

    本文介绍了一种新的训练方法，使用负反馈机制来增强DNN模型的鲁棒性，特别是在存在设备变异的情况下。

    

    利用非挥发性存储器(NVM)实现的内存计算(CiM)为加速深度神经网络(DNNs)提供了一种高效的方法。 CiM加速器通过在同一电路板结构中存储网络权重和执行矩阵操作，以最小的面积需求和异常的能效，提供DNN推理加速。然而，NVM设备的随机性和内在变化往往导致性能降低，如与预期结果相比减少分类精度。尽管提出了几种方法来减轻设备变异并增强鲁棒性，但大多数方法都依赖于整体调节并缺乏对训练过程的限制。受到负反馈机制的启发，我们引入了一种新的训练方法，使用多出口机制作为负反馈，在设备变异的情况下增强DNN模型的性能。

    Compute-in-Memory (CiM) utilizing non-volatile memory (NVM) devices presents a highly promising and efficient approach for accelerating deep neural networks (DNNs). By concurrently storing network weights and performing matrix operations within the same crossbar structure, CiM accelerators offer DNN inference acceleration with minimal area requirements and exceptional energy efficiency. However, the stochasticity and intrinsic variations of NVM devices often lead to performance degradation, such as reduced classification accuracy, compared to expected outcomes. Although several methods have been proposed to mitigate device variation and enhance robustness, most of them rely on overall modulation and lack constraints on the training process. Drawing inspiration from the negative feedback mechanism, we introduce a novel training approach that uses a multi-exit mechanism as negative feedback to enhance the performance of DNN models in the presence of device variation. Our negative feedbac
    

