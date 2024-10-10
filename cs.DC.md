# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Byzantine-Resilience of Distillation-Based Federated Learning](https://arxiv.org/abs/2402.12265) | 基于蒸馏的联邦学习在拜占庭环境下表现出极强的弹性，介绍了两种新的拜占庭攻击，并提出了一种增强拜占庭弹性的新方法。 |
| [^2] | [Federated learning with distributed fixed design quantum chips and quantum channels.](http://arxiv.org/abs/2401.13421) | 本论文提出了一种具有分布式固定设计量子芯片和量子信道的量子联邦学习模型，通过量子态的传递和聚合梯度来更新参数，提供更高的隐私保护和指数级的效率。 |

# 详细

[^1]: 论基于蒸馏的联邦学习在拜占庭环境下的弹性

    On the Byzantine-Resilience of Distillation-Based Federated Learning

    [https://arxiv.org/abs/2402.12265](https://arxiv.org/abs/2402.12265)

    基于蒸馏的联邦学习在拜占庭环境下表现出极强的弹性，介绍了两种新的拜占庭攻击，并提出了一种增强拜占庭弹性的新方法。

    

    由于在隐私、非独立同分布数据和通信成本方面的优势，使用知识蒸馏（KD）的联邦学习（FL）算法受到越来越多的关注。本文研究了这些方法在拜占庭环境中的性能，展示了基于KD的FL算法相当具有弹性，并分析了拜占庭客户端如何影响学习过程相对于联邦平均算法。根据这些见解，我们介绍了两种新的拜占庭攻击，并证明它们对先前的拜占庭弹性方法是有效的。此外，我们提出了FilterExp，一种旨在增强拜占庭弹性的新方法。

    arXiv:2402.12265v1 Announce Type: cross  Abstract: Federated Learning (FL) algorithms using Knowledge Distillation (KD) have received increasing attention due to their favorable properties with respect to privacy, non-i.i.d. data and communication cost. These methods depart from transmitting model parameters and, instead, communicate information about a learning task by sharing predictions on a public dataset. In this work, we study the performance of such approaches in the byzantine setting, where a subset of the clients act in an adversarial manner aiming to disrupt the learning process. We show that KD-based FL algorithms are remarkably resilient and analyze how byzantine clients can influence the learning process compared to Federated Averaging. Based on these insights, we introduce two new byzantine attacks and demonstrate that they are effective against prior byzantine-resilient methods. Additionally, we propose FilterExp, a novel method designed to enhance the byzantine resilien
    
[^2]: 具有分布式固定设计量子芯片和量子信道的联邦学习

    Federated learning with distributed fixed design quantum chips and quantum channels. (arXiv:2401.13421v1 [quant-ph])

    [http://arxiv.org/abs/2401.13421](http://arxiv.org/abs/2401.13421)

    本论文提出了一种具有分布式固定设计量子芯片和量子信道的量子联邦学习模型，通过量子态的传递和聚合梯度来更新参数，提供更高的隐私保护和指数级的效率。

    

    经过客户端的精心设计查询，经典联邦学习中的隐私可以被突破。然而，由于数据中的测量会导致信息的丢失，量子通信信道被认为更加安全，因为可以检测到这种信息丢失。因此，量子版本的联邦学习可以提供更多的隐私保护。此外，通过量子信道发送N维数据向量需要发送log N个纠缠态量子比特，如果数据向量作为量子态获取，这可以提供指数级的效率。在本文中，我们提出了一种量子联邦学习模型，其中基于由集中式服务器发送的量子态，操作固定设计的量子芯片。基于接收到的叠加态，客户端计算并将其本地梯度作为量子态发送到服务器，服务器将这些梯度聚合以更新参数。由于服务器不发送模型信息，

    The privacy in classical federated learning can be breached through the use of local gradient results by using engineered queries from the clients. However, quantum communication channels are considered more secure because the use of measurements in the data causes some loss of information, which can be detected. Therefore, the quantum version of federated learning can be used to provide more privacy. Additionally, sending an $N$ dimensional data vector through a quantum channel requires sending $\log N$ entangled qubits, which can provide exponential efficiency if the data vector is obtained as quantum states.  In this paper, we propose a quantum federated learning model where fixed design quantum chips are operated based on the quantum states sent by a centralized server. Based on the coming superposition states, the clients compute and then send their local gradients as quantum states to the server, where they are aggregated to update parameters. Since the server does not send model
    

