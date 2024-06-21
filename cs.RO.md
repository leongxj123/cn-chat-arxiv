# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [HumanoidBench: Simulated Humanoid Benchmark for Whole-Body Locomotion and Manipulation](https://arxiv.org/abs/2403.10506) | 提出了一个高维度的仿真机器人学习基准测试HumanoidBench，揭示了目前最先进的强化学习算法在大多数任务上面临挑战，而具备鲁棒低级策略支持的分层学习基线表现更优秀。 |
| [^2] | [MapGPT: Map-Guided Prompting with Adaptive Path Planning for Vision-and-Language Navigation](https://arxiv.org/abs/2401.07314) | MapGPT引入了在线语言形成的地图，帮助GPT理解整体环境，提出自适应规划机制以协助代理执行多步路径规划。 |
| [^3] | [M2CURL: Sample-Efficient Multimodal Reinforcement Learning via Self-Supervised Representation Learning for Robotic Manipulation.](http://arxiv.org/abs/2401.17032) | M2CURL是一种样本高效的多模态强化学习方法，通过自监督表示学习从视触觉数据中学习出高效的表示，并加速强化学习算法的收敛。 |
| [^4] | [A Fast and Optimal Learning-based Path Planning Method for Planetary Rovers.](http://arxiv.org/abs/2308.04792) | 本文提出了一种基于学习的快速路径规划方法，通过学习最优路径示范中的语义信息和地图表示，生成概率分布来搜索最优路径。实验结果表明，该方法能够提高行星探测车的探索效率。 |
| [^5] | [Simulating the Integration of Urban Air Mobility into Existing Transportation Systems: A Survey.](http://arxiv.org/abs/2301.12901) | 本文调查了城市空中出行（UAM）在大都市交通中的研究现状，确定了将UAM融入城市交通系统的关键挑战和机遇，包括对现有交通模式和拥堵的影响；安全分析和风险评估；潜在的经济和环境效益；以及为UAM和地面交通开发共享基础设施和路线。同时，我们讨论了UAM的潜在好处，如缩短旅行时间和改善服务不足地区的可达性。 |

# 详细

[^1]: HumanoidBench：用于全身运动和操作的仿真人型机器人基准测试

    HumanoidBench: Simulated Humanoid Benchmark for Whole-Body Locomotion and Manipulation

    [https://arxiv.org/abs/2403.10506](https://arxiv.org/abs/2403.10506)

    提出了一个高维度的仿真机器人学习基准测试HumanoidBench，揭示了目前最先进的强化学习算法在大多数任务上面临挑战，而具备鲁棒低级策略支持的分层学习基线表现更优秀。

    

    人型机器人在协助人类在不同环境和任务中有着巨大潜力，由于其灵活性和适应性，可以利用类人形态。然而，人型机器人的研究常常受到昂贵且易损的硬件设置的限制。为了加速人型机器人算法研究，我们提出了一个高维度的仿真机器人学习基准测试，HumanoidBench，该测试包括一个配备灵巧手部和各种具有挑战性的全身操作和运动任务的人型机器人。我们的研究发现表明，最先进的强化学习算法在大多数任务上表现不佳，而具备鲁棒的低级策略支持的分层学习基线在行走或到达等任务中表现优异。借助HumanoidBench，我们为机器人社区提供了一个平台，用于识别解决人型机器人在解决各种任务时面临的挑战，促进算法研究。

    arXiv:2403.10506v1 Announce Type: cross  Abstract: Humanoid robots hold great promise in assisting humans in diverse environments and tasks, due to their flexibility and adaptability leveraging human-like morphology. However, research in humanoid robots is often bottlenecked by the costly and fragile hardware setups. To accelerate algorithmic research in humanoid robots, we present a high-dimensional, simulated robot learning benchmark, HumanoidBench, featuring a humanoid robot equipped with dexterous hands and a variety of challenging whole-body manipulation and locomotion tasks. Our findings reveal that state-of-the-art reinforcement learning algorithms struggle with most tasks, whereas a hierarchical learning baseline achieves superior performance when supported by robust low-level policies, such as walking or reaching. With HumanoidBench, we provide the robotics community with a platform to identify the challenges arising when solving diverse tasks with humanoid robots, facilitatin
    
[^2]: MapGPT：具有自适应路径规划的地图引导提示的视觉与语言导航

    MapGPT: Map-Guided Prompting with Adaptive Path Planning for Vision-and-Language Navigation

    [https://arxiv.org/abs/2401.07314](https://arxiv.org/abs/2401.07314)

    MapGPT引入了在线语言形成的地图，帮助GPT理解整体环境，提出自适应规划机制以协助代理执行多步路径规划。

    

    具有GPT作为大脑的体验代理表现出在各种任务中的非凡决策和泛化能力。然而，现有的视觉与语言导航（VLN）零-shot代理只促使GPT-4在局部环境中选择潜在位置，而没有为代理构建一个有效的“全局视图”来理解整体环境。在这项工作中，我们提出了一种新颖的地图引导的基于GPT的代理，名为MapGPT，它引入了一个在线语言形成的地图来鼓励全局探索。具体而言，我们构建了一个在线地图，并将其合并到包含节点信息和拓扑关系的提示中，以帮助GPT理解空间环境。从这一设计中获益，我们进一步提出了一种自适应规划机制，以帮助代理根据地图执行多步规划，系统地探索多个候选

    arXiv:2401.07314v2 Announce Type: replace  Abstract: Embodied agents equipped with GPT as their brain have exhibited extraordinary decision-making and generalization abilities across various tasks. However, existing zero-shot agents for vision-and-language navigation (VLN) only prompt the GPT-4 to select potential locations within localized environments, without constructing an effective "global-view" for the agent to understand the overall environment. In this work, we present a novel map-guided GPT-based agent, dubbed MapGPT, which introduces an online linguistic-formed map to encourage the global exploration. Specifically, we build an online map and incorporate it into the prompts that include node information and topological relationships, to help GPT understand the spatial environment. Benefiting from this design, we further propose an adaptive planning mechanism to assist the agent in performing multi-step path planning based on a map, systematically exploring multiple candidate 
    
[^3]: M2CURL: 通过自监督表示学习实现样本高效的多模态强化学习，用于机器人操纵

    M2CURL: Sample-Efficient Multimodal Reinforcement Learning via Self-Supervised Representation Learning for Robotic Manipulation. (arXiv:2401.17032v1 [cs.RO])

    [http://arxiv.org/abs/2401.17032](http://arxiv.org/abs/2401.17032)

    M2CURL是一种样本高效的多模态强化学习方法，通过自监督表示学习从视触觉数据中学习出高效的表示，并加速强化学习算法的收敛。

    

    多模态强化学习中最重要的方面之一是有效地整合不同的观测模态。从这些模态中得到稳健准确的表示对于提升强化学习算法的鲁棒性和样本效率至关重要。然而，在视触觉数据的强化学习环境中学习表示面临着重要挑战，特别是由于数据的高维度和将视触觉输入与动态环境和任务目标进行相关性分析的复杂性。为了解决这些挑战，我们提出了多模态对比无监督强化学习（M2CURL）。我们的方法采用了一种新颖的多模态自监督学习技术，学习出高效的表示并加速了强化学习算法的收敛。我们的方法与强化学习算法无关，因此可以与任何可用的强化学习算法进行整合。我们在Tactile Gym 2模拟器上评估了M2CURL。

    One of the most critical aspects of multimodal Reinforcement Learning (RL) is the effective integration of different observation modalities. Having robust and accurate representations derived from these modalities is key to enhancing the robustness and sample efficiency of RL algorithms. However, learning representations in RL settings for visuotactile data poses significant challenges, particularly due to the high dimensionality of the data and the complexity involved in correlating visual and tactile inputs with the dynamic environment and task objectives. To address these challenges, we propose Multimodal Contrastive Unsupervised Reinforcement Learning (M2CURL). Our approach employs a novel multimodal self-supervised learning technique that learns efficient representations and contributes to faster convergence of RL algorithms. Our method is agnostic to the RL algorithm, thus enabling its integration with any available RL algorithm. We evaluate M2CURL on the Tactile Gym 2 simulator 
    
[^4]: 快速和最优的基于学习的行星探测车路径规划方法

    A Fast and Optimal Learning-based Path Planning Method for Planetary Rovers. (arXiv:2308.04792v1 [cs.RO])

    [http://arxiv.org/abs/2308.04792](http://arxiv.org/abs/2308.04792)

    本文提出了一种基于学习的快速路径规划方法，通过学习最优路径示范中的语义信息和地图表示，生成概率分布来搜索最优路径。实验结果表明，该方法能够提高行星探测车的探索效率。

    

    智能自主路径规划对于提高行星探测车的探索效率至关重要。在本文中，我们提出了一种基于学习的方法，在高程图中快速搜索最优路径，称为NNPP。NNPP模型从大量预注释的最优路径示范中学习起始和目标位置的语义信息，以及地图表示，并生成每个像素的概率分布，表示其属于地图上最优路径的可能性。具体而言，本文从DEM获取的坡度、粗糙度和高度差计算每个网格单元的遍历成本。随后，使用高斯分布对起始和目标位置进行编码，并分析不同位置编码参数对模型性能的影响。经过训练，NNPP模型能够在新的地图上执行路径规划。实验证明，NNPP生成的引导场能够准确指导行星探测车的运动。

    Intelligent autonomous path planning is crucial to improve the exploration efficiency of planetary rovers. In this paper, we propose a learning-based method to quickly search for optimal paths in an elevation map, which is called NNPP. The NNPP model learns semantic information about start and goal locations, as well as map representations, from numerous pre-annotated optimal path demonstrations, and produces a probabilistic distribution over each pixel representing the likelihood of it belonging to an optimal path on the map. More specifically, the paper computes the traversal cost for each grid cell from the slope, roughness and elevation difference obtained from the DEM. Subsequently, the start and goal locations are encoded using a Gaussian distribution and different location encoding parameters are analyzed for their effect on model performance. After training, the NNPP model is able to perform path planning on novel maps. Experiments show that the guidance field generated by the 
    
[^5]: 模拟城市空中出行融入现有交通系统：一项调查

    Simulating the Integration of Urban Air Mobility into Existing Transportation Systems: A Survey. (arXiv:2301.12901v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2301.12901](http://arxiv.org/abs/2301.12901)

    本文调查了城市空中出行（UAM）在大都市交通中的研究现状，确定了将UAM融入城市交通系统的关键挑战和机遇，包括对现有交通模式和拥堵的影响；安全分析和风险评估；潜在的经济和环境效益；以及为UAM和地面交通开发共享基础设施和路线。同时，我们讨论了UAM的潜在好处，如缩短旅行时间和改善服务不足地区的可达性。

    This paper surveys the current state of research on urban air mobility (UAM) in metropolitan-scale traffic using simulation techniques, identifying key challenges and opportunities for integrating UAM into urban transportation systems, including impacts on existing traffic patterns and congestion, safety analysis and risk assessment, potential economic and environmental benefits, and the development of shared infrastructure and routes for UAM and ground-based transportation. The potential benefits of UAM, such as reduced travel times and improved accessibility for underserved areas, are also discussed.

    城市空中出行（UAM）有可能彻底改变大都市地区的交通方式，提供一种新的交通方式，缓解拥堵，提高可达性。然而，将UAM融入现有交通系统是一项复杂的任务，需要深入了解其对交通流量和容量的影响。在本文中，我们进行了一项调查，使用模拟技术调查了UAM在大都市交通中的研究现状。我们确定了将UAM融入城市交通系统的关键挑战和机遇，包括对现有交通模式和拥堵的影响；安全分析和风险评估；潜在的经济和环境效益；以及为UAM和地面交通开发共享基础设施和路线。我们还讨论了UAM的潜在好处，如缩短旅行时间和改善服务不足地区的可达性。我们的调查

    Urban air mobility (UAM) has the potential to revolutionize transportation in metropolitan areas, providing a new mode of transportation that could alleviate congestion and improve accessibility. However, the integration of UAM into existing transportation systems is a complex task that requires a thorough understanding of its impact on traffic flow and capacity. In this paper, we conduct a survey to investigate the current state of research on UAM in metropolitan-scale traffic using simulation techniques. We identify key challenges and opportunities for the integration of UAM into urban transportation systems, including impacts on existing traffic patterns and congestion; safety analysis and risk assessment; potential economic and environmental benefits; and the development of shared infrastructure and routes for UAM and ground-based transportation. We also discuss the potential benefits of UAM, such as reduced travel times and improved accessibility for underserved areas. Our survey 
    

