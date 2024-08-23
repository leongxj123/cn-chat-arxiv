# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Policy Learning via Offline Skill Diffusion](https://arxiv.org/abs/2403.00225) | 提出了一种新颖的离线技能学习框架DuSkill，通过引导扩散模型生成通用技能，从而增强不同领域任务的策略学习鲁棒性。 |
| [^2] | [CGGM: A conditional graph generation model with adaptive sparsity for node anomaly detection in IoT networks](https://arxiv.org/abs/2402.17363) | CGGM是一种新颖的图生成模型，通过自适应稀疏性生成邻接矩阵，解决了物联网网络中节点异常检测中节点类别不平衡的问题 |
| [^3] | [Pointing the Way: Refining Radar-Lidar Localization Using Learned ICP Weights.](http://arxiv.org/abs/2309.08731) | 本文提出了一种深度学习方法，通过学习到的ICP权重优化雷达-激光雷达的定位，从而改善了雷达测量对激光雷达地图的定位效果。这一方法在保持高质量地图定位性能的同时，提高了在降水和大雾等恶劣天气条件下的定位准确性。 |

# 详细

[^1]: 通过离线技能扩散实现稳健策略学习

    Robust Policy Learning via Offline Skill Diffusion

    [https://arxiv.org/abs/2403.00225](https://arxiv.org/abs/2403.00225)

    提出了一种新颖的离线技能学习框架DuSkill，通过引导扩散模型生成通用技能，从而增强不同领域任务的策略学习鲁棒性。

    

    基于技能的强化学习方法在解决长时域任务中表现出了相当大的潜力，尤其是通过分层结构。这些技能是从离线数据集中无关任务地学习的，可以加快针对新任务的策略学习过程。然而，由于这些技能在不同领域中的应用仍受限于对数据集的固有依赖，当尝试通过强化学习为不同于数据集领域的目标领域学习基于技能的策略时，这一挑战就变得困难。在本文中，我们提出了一个新颖的离线技能学习框架DuSkill，它采用了引导扩散模型来生成从数据集中有限技能扩展出的通用技能，从而增强了不同领域任务的策略学习鲁棒性。具体来说，我们设计了一个引导扩散技能解码器，结合分层编码，以解开技能嵌入。

    arXiv:2403.00225v1 Announce Type: new  Abstract: Skill-based reinforcement learning (RL) approaches have shown considerable promise, especially in solving long-horizon tasks via hierarchical structures. These skills, learned task-agnostically from offline datasets, can accelerate the policy learning process for new tasks. Yet, the application of these skills in different domains remains restricted due to their inherent dependency on the datasets, which poses a challenge when attempting to learn a skill-based policy via RL for a target domain different from the datasets' domains. In this paper, we present a novel offline skill learning framework DuSkill which employs a guided Diffusion model to generate versatile skills extended from the limited skills in datasets, thereby enhancing the robustness of policy learning for tasks in different domains. Specifically, we devise a guided diffusion-based skill decoder in conjunction with the hierarchical encoding to disentangle the skill embeddi
    
[^2]: CGGM：一种具有自适应稀疏性的条件图生成模型，用于物联网网络中节点异常检测

    CGGM: A conditional graph generation model with adaptive sparsity for node anomaly detection in IoT networks

    [https://arxiv.org/abs/2402.17363](https://arxiv.org/abs/2402.17363)

    CGGM是一种新颖的图生成模型，通过自适应稀疏性生成邻接矩阵，解决了物联网网络中节点异常检测中节点类别不平衡的问题

    

    动态图被广泛用于检测物联网中节点的异常行为。生成模型通常用于解决动态图中节点类别不平衡的问题。然而，它面临的约束包括邻接关系的单调性，为节点构建多维特征的困难，以及缺乏端到端生成多类节点的方法。本文提出了一种名为CGGM的新颖图生成模型，专门设计用于生成少数类别中更多节点。通过自适应稀疏性生成邻接矩阵的机制增强了其结构的灵活性。特征生成模块名为多维特征生成器（MFG），可生成包括拓扑信息在内的节点特征。标签被转换为嵌入向量，用作条件。

    arXiv:2402.17363v1 Announce Type: cross  Abstract: Dynamic graphs are extensively employed for detecting anomalous behavior in nodes within the Internet of Things (IoT). Generative models are often used to address the issue of imbalanced node categories in dynamic graphs. Nevertheless, the constraints it faces include the monotonicity of adjacency relationships, the difficulty in constructing multi-dimensional features for nodes, and the lack of a method for end-to-end generation of multiple categories of nodes. This paper presents a novel graph generation model, called CGGM, designed specifically to generate a larger number of nodes belonging to the minority class. The mechanism for generating an adjacency matrix, through adaptive sparsity, enhances flexibility in its structure. The feature generation module, called multidimensional features generator (MFG) to generate node features along with topological information. Labels are transformed into embedding vectors, serving as condition
    
[^3]: 指引的方法：利用学习到的ICP权重改进雷达-激光雷达定位

    Pointing the Way: Refining Radar-Lidar Localization Using Learned ICP Weights. (arXiv:2309.08731v1 [cs.RO])

    [http://arxiv.org/abs/2309.08731](http://arxiv.org/abs/2309.08731)

    本文提出了一种深度学习方法，通过学习到的ICP权重优化雷达-激光雷达的定位，从而改善了雷达测量对激光雷达地图的定位效果。这一方法在保持高质量地图定位性能的同时，提高了在降水和大雾等恶劣天气条件下的定位准确性。

    

    本文提出了一种基于深度学习的新方法，用于改进雷达测量对激光雷达地图的定位。虽然目前定位的技术水平是将激光雷达数据与激光雷达地图进行匹配，但是雷达被认为是一种有前途的替代方法，因为它对降水和大雾等恶劣天气具有更强的韧性。为了利用现有的高质量激光雷达地图，同时在恶劣天气下保持性能，将雷达数据与激光雷达地图进行匹配具有重要意义。然而，由于雷达测量中存在的独特伪影，雷达-激光雷达定位一直难以达到与激光雷达-激光雷达系统相媲美的性能，使其无法用于自动驾驶。本工作在基于ICP的雷达-激光雷达定位系统基础上，包括一个学习的预处理步骤，根据高层次的扫描信息对雷达点进行加权。将经过验证的分析方法与学习到的权重相结合，减小了雷达定位中的误差。

    This paper presents a novel deep-learning-based approach to improve localizing radar measurements against lidar maps. Although the state of the art for localization is matching lidar data to lidar maps, radar has been considered as a promising alternative, as it is potentially more resilient against adverse weather such as precipitation and heavy fog. To make use of existing high-quality lidar maps, while maintaining performance in adverse weather, matching radar data to lidar maps is of interest. However, owing in part to the unique artefacts present in radar measurements, radar-lidar localization has struggled to achieve comparable performance to lidar-lidar systems, preventing it from being viable for autonomous driving. This work builds on an ICP-based radar-lidar localization system by including a learned preprocessing step that weights radar points based on high-level scan information. Combining a proven analytical approach with a learned weight reduces localization errors in rad
    

