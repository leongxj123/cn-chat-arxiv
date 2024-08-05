# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DASA: Delay-Adaptive Multi-Agent Stochastic Approximation](https://arxiv.org/abs/2403.17247) | DASA算法是第一个收敛速度仅依赖于混合时间和平均延迟的算法，同时在马尔科夫采样下实现N倍的收敛加速。 |
| [^2] | [Towards Model-Free LQR Control over Rate-Limited Channels.](http://arxiv.org/abs/2401.01258) | 这篇论文研究了在速率限制通道上实现模型无关的LQR控制的问题。通过引入自适应量化梯度下降（AQGD）算法，作者证明了在噪声电路中可以实现控制问题的解决。 |
| [^3] | [Unsupervised Graph-based Learning Method for Sub-band Allocation in 6G Subnetworks.](http://arxiv.org/abs/2401.00950) | 本文提出了一种无监督的基于图的学习方法，用于在6G子网络中进行子频带分配。该方法通过优化使用图神经网络的子频带分配，实现了与集中式贪婪着色子频带分配方法相近的性能，并且具有更低的计算时间复杂度和较小的信令开销。 |

# 详细

[^1]: DASA: 延迟自适应多智能体随机逼近

    DASA: Delay-Adaptive Multi-Agent Stochastic Approximation

    [https://arxiv.org/abs/2403.17247](https://arxiv.org/abs/2403.17247)

    DASA算法是第一个收敛速度仅依赖于混合时间和平均延迟的算法，同时在马尔科夫采样下实现N倍的收敛加速。

    

    我们考虑一种设置，其中$N$个智能体旨在通过并行操作并与中央服务器通信来加速一个常见的随机逼近（SA）问题。我们假定上行传输到服务器的传输受到异步和潜在无界时变延迟的影响。为了减轻延迟和落后者的影响，同时又能获得分布式计算的好处，我们提出了一种名为DASA的延迟自适应多智能体随机逼近算法。我们对DASA进行了有限时间分析，假设智能体的随机观测过程是独立马尔科夫链。与现有结果相比，DASA是第一个其收敛速度仅取决于混合时间$tmix$和平均延迟$\tau_{avg}$，同时在马尔科夫采样下实现N倍的收敛加速的算法。我们的工作对于各种SA应用是相关的。

    arXiv:2403.17247v1 Announce Type: new  Abstract: We consider a setting in which $N$ agents aim to speedup a common Stochastic Approximation (SA) problem by acting in parallel and communicating with a central server. We assume that the up-link transmissions to the server are subject to asynchronous and potentially unbounded time-varying delays. To mitigate the effect of delays and stragglers while reaping the benefits of distributed computation, we propose \texttt{DASA}, a Delay-Adaptive algorithm for multi-agent Stochastic Approximation. We provide a finite-time analysis of \texttt{DASA} assuming that the agents' stochastic observation processes are independent Markov chains. Significantly advancing existing results, \texttt{DASA} is the first algorithm whose convergence rate depends only on the mixing time $\tmix$ and on the average delay $\tau_{avg}$ while jointly achieving an $N$-fold convergence speedup under Markovian sampling. Our work is relevant for various SA applications, inc
    
[^2]: 实现模型无关的通过速率限制通道的LQR控制

    Towards Model-Free LQR Control over Rate-Limited Channels. (arXiv:2401.01258v1 [math.OC])

    [http://arxiv.org/abs/2401.01258](http://arxiv.org/abs/2401.01258)

    这篇论文研究了在速率限制通道上实现模型无关的LQR控制的问题。通过引入自适应量化梯度下降（AQGD）算法，作者证明了在噪声电路中可以实现控制问题的解决。

    

    鉴于模型无关方法在许多问题设置中的控制设计方面取得的成功，自然而然地会问，如果利用实际的通信通道来传输梯度或策略，情况会如何改变。尽管由此产生的问题与网络控制系统中研究的公式有类似之处，但那个领域的丰富文献通常假定系统的模型是已知的。为了在模型无关控制设计和网络控制系统领域之间建立联系，我们提出了一个问题：\textit{是否可以通过速率限制的通道以模型无关的方式解决基本的控制问题-例如线性二次调节器（LQR）问题？}为了回答这个问题，我们研究了一个设置，其中一个工作代理通过一个无噪声信道以有限的位速率传输量化策略梯度（LQR成本）到一个服务器。我们提出了一种名为自适应量化梯度下降（AQGD）的新算法，并证明了

    Given the success of model-free methods for control design in many problem settings, it is natural to ask how things will change if realistic communication channels are utilized for the transmission of gradients or policies. While the resulting problem has analogies with the formulations studied under the rubric of networked control systems, the rich literature in that area has typically assumed that the model of the system is known. As a step towards bridging the fields of model-free control design and networked control systems, we ask: \textit{Is it possible to solve basic control problems - such as the linear quadratic regulator (LQR) problem - in a model-free manner over a rate-limited channel?} Toward answering this question, we study a setting where a worker agent transmits quantized policy gradients (of the LQR cost) to a server over a noiseless channel with a finite bit-rate. We propose a new algorithm titled Adaptively Quantized Gradient Descent (\texttt{AQGD}), and prove that
    
[^3]: 无监督的基于图的学习方法用于6G子网络的子频带分配

    Unsupervised Graph-based Learning Method for Sub-band Allocation in 6G Subnetworks. (arXiv:2401.00950v1 [cs.NI])

    [http://arxiv.org/abs/2401.00950](http://arxiv.org/abs/2401.00950)

    本文提出了一种无监督的基于图的学习方法，用于在6G子网络中进行子频带分配。该方法通过优化使用图神经网络的子频带分配，实现了与集中式贪婪着色子频带分配方法相近的性能，并且具有更低的计算时间复杂度和较小的信令开销。

    

    在本文中，我们提出了一种无监督的基于图的学习方法，用于在无线网络中进行频率子带分配。我们考虑在工厂环境中密集部署的子网络，这些子网络只有有限数量的子频带，必须被优化地分配以协调子网络间的干扰。我们将子网络部署建模为一个冲突图，并提出了一种受到图着色启发和Potts模型的无监督学习方法，利用图神经网络来优化子频带分配。数值评估表明，所提出的方法在较低的计算时间复杂度下，实现了与集中式贪婪着色子频带分配启发式方法接近的性能。此外，与需要所有互相干扰的信道信息的迭代优化启发式相比，它产生更少的信令开销。我们进一步证明该方法对不同的网络设置具有健壮性。

    In this paper, we present an unsupervised approach for frequency sub-band allocation in wireless networks using graph-based learning. We consider a dense deployment of subnetworks in the factory environment with a limited number of sub-bands which must be optimally allocated to coordinate inter-subnetwork interference. We model the subnetwork deployment as a conflict graph and propose an unsupervised learning approach inspired by the graph colouring heuristic and the Potts model to optimize the sub-band allocation using graph neural networks. The numerical evaluation shows that the proposed method achieves close performance to the centralized greedy colouring sub-band allocation heuristic with lower computational time complexity. In addition, it incurs reduced signalling overhead compared to iterative optimization heuristics that require all the mutual interfering channel information. We further demonstrate that the method is robust to different network settings.
    

