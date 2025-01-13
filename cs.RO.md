# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GUTS: Generalized Uncertainty-Aware Thompson Sampling for Multi-Agent Active Search.](http://arxiv.org/abs/2304.02075) | 本文提出了 GUTS 算法，能够在异构多机器人系统中部署，解决大型非结构化环境下的主动搜索问题，考虑了传感器不确定性、遮挡问题、异构搜索团队和硬件、通信故障的鲁棒性等问题，并在仿真中超过现有算法高达80%。 |

# 详细

[^1]: GUTS：多智能体主动搜索的广义不确定性感知 Thompson Sampling算法

    GUTS: Generalized Uncertainty-Aware Thompson Sampling for Multi-Agent Active Search. (arXiv:2304.02075v1 [cs.RO])

    [http://arxiv.org/abs/2304.02075](http://arxiv.org/abs/2304.02075)

    本文提出了 GUTS 算法，能够在异构多机器人系统中部署，解决大型非结构化环境下的主动搜索问题，考虑了传感器不确定性、遮挡问题、异构搜索团队和硬件、通信故障的鲁棒性等问题，并在仿真中超过现有算法高达80%。

    

    快速灾难响应的机器人解决方案对确保最小生命损失至关重要，特别是当搜索区域对于人类救援者而言过于危险或过于广阔时。本文将这个问题建模为一个异步多智能体主动搜索任务，在此任务中，每个机器人旨在高效地在未知环境中寻找感兴趣的对象（OOI）。这种表述解决了搜索任务应该专注于快速恢复OOI而不是对搜索区域进行全覆盖的要求。先前的方法未能准确建模传感器不确定性，考虑到由于植被或地形的遮挡而导致的遮挡问题，或者考虑到异构搜索团队和硬件、通信故障的鲁棒性要求。我们提出了广义不确定性感知Thompson抽样（GUTS）算法，它解决了这些问题，并适用于在大型非结构化环境中部署异构多机器人系统进行主动搜索。我们通过仿真实验表明，GUTS算法在性能上超过现有算法高达80%。

    Robotic solutions for quick disaster response are essential to ensure minimal loss of life, especially when the search area is too dangerous or too vast for human rescuers. We model this problem as an asynchronous multi-agent active-search task where each robot aims to efficiently seek objects of interest (OOIs) in an unknown environment. This formulation addresses the requirement that search missions should focus on quick recovery of OOIs rather than full coverage of the search region. Previous approaches fail to accurately model sensing uncertainty, account for occlusions due to foliage or terrain, or consider the requirement for heterogeneous search teams and robustness to hardware and communication failures. We present the Generalized Uncertainty-aware Thompson Sampling (GUTS) algorithm, which addresses these issues and is suitable for deployment on heterogeneous multi-robot systems for active search in large unstructured environments. We show through simulation experiments that GU
    

