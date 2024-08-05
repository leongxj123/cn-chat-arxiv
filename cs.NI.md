# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unsupervised Graph-based Learning Method for Sub-band Allocation in 6G Subnetworks.](http://arxiv.org/abs/2401.00950) | 本文提出了一种无监督的基于图的学习方法，用于在6G子网络中进行子频带分配。该方法通过优化使用图神经网络的子频带分配，实现了与集中式贪婪着色子频带分配方法相近的性能，并且具有更低的计算时间复杂度和较小的信令开销。 |
| [^2] | [Insights from the Design Space Exploration of Flow-Guided Nanoscale Localization.](http://arxiv.org/abs/2305.18493) | 研究了基于流导向纳米定位的设计空间，考虑了能源和信号衰减等因素，为这一新兴领域提供了有希望的解决方案。 |

# 详细

[^1]: 无监督的基于图的学习方法用于6G子网络的子频带分配

    Unsupervised Graph-based Learning Method for Sub-band Allocation in 6G Subnetworks. (arXiv:2401.00950v1 [cs.NI])

    [http://arxiv.org/abs/2401.00950](http://arxiv.org/abs/2401.00950)

    本文提出了一种无监督的基于图的学习方法，用于在6G子网络中进行子频带分配。该方法通过优化使用图神经网络的子频带分配，实现了与集中式贪婪着色子频带分配方法相近的性能，并且具有更低的计算时间复杂度和较小的信令开销。

    

    在本文中，我们提出了一种无监督的基于图的学习方法，用于在无线网络中进行频率子带分配。我们考虑在工厂环境中密集部署的子网络，这些子网络只有有限数量的子频带，必须被优化地分配以协调子网络间的干扰。我们将子网络部署建模为一个冲突图，并提出了一种受到图着色启发和Potts模型的无监督学习方法，利用图神经网络来优化子频带分配。数值评估表明，所提出的方法在较低的计算时间复杂度下，实现了与集中式贪婪着色子频带分配启发式方法接近的性能。此外，与需要所有互相干扰的信道信息的迭代优化启发式相比，它产生更少的信令开销。我们进一步证明该方法对不同的网络设置具有健壮性。

    In this paper, we present an unsupervised approach for frequency sub-band allocation in wireless networks using graph-based learning. We consider a dense deployment of subnetworks in the factory environment with a limited number of sub-bands which must be optimally allocated to coordinate inter-subnetwork interference. We model the subnetwork deployment as a conflict graph and propose an unsupervised learning approach inspired by the graph colouring heuristic and the Potts model to optimize the sub-band allocation using graph neural networks. The numerical evaluation shows that the proposed method achieves close performance to the centralized greedy colouring sub-band allocation heuristic with lower computational time complexity. In addition, it incurs reduced signalling overhead compared to iterative optimization heuristics that require all the mutual interfering channel information. We further demonstrate that the method is robust to different network settings.
    
[^2]: 基于流导向纳米定位的设计空间探索的见解

    Insights from the Design Space Exploration of Flow-Guided Nanoscale Localization. (arXiv:2305.18493v1 [cs.NI])

    [http://arxiv.org/abs/2305.18493](http://arxiv.org/abs/2305.18493)

    研究了基于流导向纳米定位的设计空间，考虑了能源和信号衰减等因素，为这一新兴领域提供了有希望的解决方案。

    

    具有太赫兹无线通信能力的纳米设备为在人类血液中进行流导向定位提供了基础。此类定位使得将所感受到的事件的位置与事件本身进行匹配成为可能，从而实现了精准医疗方面的早期和精准诊断、降低成本和侵入性。流导向定位仍处于原始阶段，只有少数论文涉及此问题。尽管如此，所提出解决方案的性能评估仍然以非标准化的方式进行，通常只考虑单一的性能指标，并忽略了在这种规模（例如，纳米器件的能量受限）和对于这种具有挑战性的环境（例如，体内太赫兹传播的严重衰减）下相关的各个方面。因此，这些评估具有低水平的真实性，并且无法以客观的方式进行比较。为了解决这个问题，我们考虑了传输能量消耗和信号衰减，对流导向纳米定位的设计空间进行了探索。我们的分析考虑了各种性能指标（例如能量消耗和定位精度）和挑战（例如身体运动和血压），导致我们可以为这个新兴领域提供有希望的解决方案。

    Nanodevices with Terahertz (THz)-based wireless communication capabilities are providing a primer for flow-guided localization within the human bloodstreams. Such localization is allowing for assigning the locations of sensed events with the events themselves, providing benefits in precision medicine along the lines of early and precise diagnostics, and reduced costs and invasiveness. Flow-guided localization is still in a rudimentary phase, with only a handful of works targeting the problem. Nonetheless, the performance assessments of the proposed solutions are already carried out in a non-standardized way, usually along a single performance metric, and ignoring various aspects that are relevant at such a scale (e.g., nanodevices' limited energy) and for such a challenging environment (e.g., extreme attenuation of in-body THz propagation). As such, these assessments feature low levels of realism and cannot be compared in an objective way. Toward addressing this issue, we account for t
    

