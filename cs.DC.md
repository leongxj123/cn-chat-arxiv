# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DISTINQT: A Distributed Privacy Aware Learning Framework for QoS Prediction for Future Mobile and Wireless Networks.](http://arxiv.org/abs/2401.10158) | DISTINQT是一种面向未来移动和无线网络的隐私感知分布式学习框架，用于QoS预测。 |
| [^2] | [FLEdge: Benchmarking Federated Machine Learning Applications in Edge Computing Systems.](http://arxiv.org/abs/2306.05172) | FLEdge是一个面向边缘计算系统中FL工作量的基准测试，通过研究硬件异构性、能量效率和隐私级别对FL系统训练的影响，以及客户端退出对最新FL策略的影响，提供了训练最先进的FL工作负载的新见解。 |

# 详细

[^1]: DISTINQT: 一种面向未来移动和无线网络的分布式隐私感知学习框架，用于QoS预测

    DISTINQT: A Distributed Privacy Aware Learning Framework for QoS Prediction for Future Mobile and Wireless Networks. (arXiv:2401.10158v1 [cs.NI])

    [http://arxiv.org/abs/2401.10158](http://arxiv.org/abs/2401.10158)

    DISTINQT是一种面向未来移动和无线网络的隐私感知分布式学习框架，用于QoS预测。

    

    5G和6G以后的网络将支持依赖一定服务质量（QoS）的新的和具有挑战性的用例和应用程序。及时预测QoS对于安全关键应用（如车辆通信）尤为重要。尽管直到最近，QoS预测一直由集中式人工智能（AI）解决方案完成，但已经出现了一些隐私、计算和运营方面的问题。替代方案已经出现（如分割学习、联邦学习），将复杂度较低的AI任务分布在节点之间，同时保护数据隐私。然而，考虑到未来无线网络的异构性，当涉及可扩展的分布式学习方法时，会出现新的挑战。该研究提出了一种名为DISTINQT的面向QoS预测的隐私感知分布式学习框架。

    Beyond 5G and 6G networks are expected to support new and challenging use cases and applications that depend on a certain level of Quality of Service (QoS) to operate smoothly. Predicting the QoS in a timely manner is of high importance, especially for safety-critical applications as in the case of vehicular communications. Although until recent years the QoS prediction has been carried out by centralized Artificial Intelligence (AI) solutions, a number of privacy, computational, and operational concerns have emerged. Alternative solutions have been surfaced (e.g. Split Learning, Federated Learning), distributing AI tasks of reduced complexity across nodes, while preserving the privacy of the data. However, new challenges rise when it comes to scalable distributed learning approaches, taking into account the heterogeneous nature of future wireless networks. The current work proposes DISTINQT, a privacy-aware distributed learning framework for QoS prediction. Our framework supports mult
    
[^2]: FLEdge：边缘计算系统中联邦机器学习应用的基准测试

    FLEdge: Benchmarking Federated Machine Learning Applications in Edge Computing Systems. (arXiv:2306.05172v1 [cs.LG])

    [http://arxiv.org/abs/2306.05172](http://arxiv.org/abs/2306.05172)

    FLEdge是一个面向边缘计算系统中FL工作量的基准测试，通过研究硬件异构性、能量效率和隐私级别对FL系统训练的影响，以及客户端退出对最新FL策略的影响，提供了训练最先进的FL工作负载的新见解。

    

    近年来，联邦机器学习（FL）备受关注。 FL基准测试主要在模拟系统或数据中心环境中进行探索，忽略了与边缘计算密切相关的实际系统设置。 我们通过引入面向边缘计算系统中FL工作量的基准测试FLEdge来弥补这一研究差距。我们系统地研究了硬件异构性、训练过程中的能量效率以及各种不同隐私级别对FL系统训练的影响。为了使这个基准测试适用于实际场景，我们评估了客户端退出对具有高达50％失效率的最新FL策略的影响。 FLEdge提供了新的见解，例如，在旧GPU加速的嵌入式设备上训练最先进的FL工作负载比在现代服务器级GPU上训练高达3倍的能量效率。

    Federated Machine Learning (FL) has received considerable attention in recent years. FL benchmarks are predominantly explored in either simulated systems or data center environments, neglecting the setups of real-world systems, which are often closely linked to edge computing. We close this research gap by introducing FLEdge, a benchmark targeting FL workloads in edge computing systems. We systematically study hardware heterogeneity, energy efficiency during training, and the effect of various differential privacy levels on training in FL systems. To make this benchmark applicable to real-world scenarios, we evaluate the impact of client dropouts on state-of-the-art FL strategies with failure rates as high as 50%. FLEdge provides new insights, such as that training state-of-the-art FL workloads on older GPU-accelerated embedded devices is up to 3x more energy efficient than on modern server-grade GPUs.
    

