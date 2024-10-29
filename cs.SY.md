# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Resource-Aware Hierarchical Federated Learning in Wireless Video Caching Networks](https://arxiv.org/abs/2402.04216) | 通过资源感知的分层联邦学习，我们提出了一种解决方案，可以预测用户未来的内容请求，并减轻无线视频缓存网络中回程流量拥塞的问题。 |
| [^2] | [Linear Convergence of Pre-Conditioned PI Consensus Algorithm under Restricted Strong Convexity.](http://arxiv.org/abs/2310.00419) | 本文在点对点多智能体网络中提出了一种使用比例积分（PI）控制策略的预条件PI共识算法，保证了其在受限强凸函数下的线性收敛性，无需个体局部代价函数的凸性，并且通过引入局部预条件进一步加速算法。 |
| [^3] | [A General Verification Framework for Dynamical and Control Models via Certificate Synthesis.](http://arxiv.org/abs/2309.06090) | 这个论文提出了一个通用的框架来通过证书合成验证动态和控制模型。研究者们提供了一种自动化方法来设计控制器并分析复杂规范。这个方法利用神经网络和SMT求解器来提供候选控制和证书函数，并为控制的安全学习领域做出了贡献。 |

# 详细

[^1]: 无线视频缓存网络中的资源感知分层联邦学习

    Resource-Aware Hierarchical Federated Learning in Wireless Video Caching Networks

    [https://arxiv.org/abs/2402.04216](https://arxiv.org/abs/2402.04216)

    通过资源感知的分层联邦学习，我们提出了一种解决方案，可以预测用户未来的内容请求，并减轻无线视频缓存网络中回程流量拥塞的问题。

    

    在无线视频缓存网络中，通过将待请求内容存储在不同级别上，可以减轻由少数热门文件的视频流量造成的回程拥塞。通常，内容服务提供商（CSP）拥有内容，用户使用其（无线）互联网服务提供商（ISP）从CSP请求其首选内容。由于这些参与方不会透露其私密信息和商业机密，传统技术可能无法用于预测用户未来需求的动态变化。出于这个原因，我们提出了一种新颖的资源感知分层联邦学习（RawHFL）解决方案，用于预测用户未来的内容请求。采用了一种实用的数据获取技术，允许用户根据其请求的内容更新其本地训练数据集。此外，由于网络和其他计算资源有限，考虑到只有一部分用户参与模型训练，我们推导出

    Backhaul traffic congestion caused by the video traffic of a few popular files can be alleviated by storing the to-be-requested content at various levels in wireless video caching networks. Typically, content service providers (CSPs) own the content, and the users request their preferred content from the CSPs using their (wireless) internet service providers (ISPs). As these parties do not reveal their private information and business secrets, traditional techniques may not be readily used to predict the dynamic changes in users' future demands. Motivated by this, we propose a novel resource-aware hierarchical federated learning (RawHFL) solution for predicting user's future content requests. A practical data acquisition technique is used that allows the user to update its local training dataset based on its requested content. Besides, since networking and other computational resources are limited, considering that only a subset of the users participate in the model training, we derive
    
[^2]: 受限强凸性下的预条件PI共识算法的线性收敛性

    Linear Convergence of Pre-Conditioned PI Consensus Algorithm under Restricted Strong Convexity. (arXiv:2310.00419v1 [math.OC])

    [http://arxiv.org/abs/2310.00419](http://arxiv.org/abs/2310.00419)

    本文在点对点多智能体网络中提出了一种使用比例积分（PI）控制策略的预条件PI共识算法，保证了其在受限强凸函数下的线性收敛性，无需个体局部代价函数的凸性，并且通过引入局部预条件进一步加速算法。

    

    本文考虑在点对点多智能体网络中解决分布式凸优化问题。网络被假定为同步和连通的。采用比例积分（PI）控制策略，开发了多种具有固定步长的算法，其中最早的是PI共识算法。利用李雅普诺夫理论，我们首次保证了具有速率匹配离散化的受限强凸函数的PI共识算法的指数收敛性，而不需要个体局部代价函数的凸性。为了加速PI共识算法，我们采用了局部预条件的形式，即常数正定矩阵，并通过数值验证其相比于突出的分布式凸优化算法的效率。

    This paper considers solving distributed convex optimization problems in peer-to-peer multi-agent networks. The network is assumed to be synchronous and connected. By using the proportional-integral (PI) control strategy, various algorithms with fixed stepsize have been developed. The earliest among them is the PI consensus algorithm. Using Lyapunov theory, we guarantee exponential convergence of the PI consensus algorithm for restricted strongly convex functions with rate-matching discretization, without requiring convexity of individual local cost functions, for the first time. In order to accelerate the PI consensus algorithm, we incorporate local pre-conditioning in the form of constant positive definite matrices and numerically validate its efficiency compared to the prominent distributed convex optimization algorithms. Unlike classical pre-conditioning, where only the gradients are multiplied by a pre-conditioner, the proposed pre-conditioning modifies both the gradients and the 
    
[^3]: 通过证书合成的动态与控制模型的通用验证框架

    A General Verification Framework for Dynamical and Control Models via Certificate Synthesis. (arXiv:2309.06090v1 [eess.SY])

    [http://arxiv.org/abs/2309.06090](http://arxiv.org/abs/2309.06090)

    这个论文提出了一个通用的框架来通过证书合成验证动态和控制模型。研究者们提供了一种自动化方法来设计控制器并分析复杂规范。这个方法利用神经网络和SMT求解器来提供候选控制和证书函数，并为控制的安全学习领域做出了贡献。

    

    控制论的一个新兴分支专门研究证书学习，涉及对自主或控制模型的所需（可能是复杂的）系统行为的规范，并通过基于函数的证明进行分析验证。然而，满足这些复杂要求的控制器的合成通常是一个非常困难的任务，可能超出了大多数专家控制工程师的能力。因此，需要自动技术能够设计控制器并分析各种复杂规范。在本文中，我们提供了一个通用框架来编码系统规范并定义相应的证书，并提出了一种自动化方法来正式合成控制器和证书。我们的方法为控制的安全学习领域做出了贡献，利用神经网络的灵活性提供候选的控制和证书函数，同时使用SMT求解器来提供形式化的保证。

    An emerging branch of control theory specialises in certificate learning, concerning the specification of a desired (possibly complex) system behaviour for an autonomous or control model, which is then analytically verified by means of a function-based proof. However, the synthesis of controllers abiding by these complex requirements is in general a non-trivial task and may elude the most expert control engineers. This results in a need for automatic techniques that are able to design controllers and to analyse a wide range of elaborate specifications. In this paper, we provide a general framework to encode system specifications and define corresponding certificates, and we present an automated approach to formally synthesise controllers and certificates. Our approach contributes to the broad field of safe learning for control, exploiting the flexibility of neural networks to provide candidate control and certificate functions, whilst using SMT-solvers to offer a formal guarantee of co
    

