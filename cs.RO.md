# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CGGM: A conditional graph generation model with adaptive sparsity for node anomaly detection in IoT networks](https://arxiv.org/abs/2402.17363) | CGGM是一种新颖的图生成模型，通过自适应稀疏性生成邻接矩阵，解决了物联网网络中节点异常检测中节点类别不平衡的问题 |
| [^2] | [Push it to the Demonstrated Limit: Multimodal Visuotactile Imitation Learning with Force Matching.](http://arxiv.org/abs/2311.01248) | 本研究利用视觉触觉传感器和模仿学习相结合，通过配对优化触觉力量曲线和简化传感器应用，对接触丰富的操作任务进行了研究。 |

# 详细

[^1]: CGGM：一种具有自适应稀疏性的条件图生成模型，用于物联网网络中节点异常检测

    CGGM: A conditional graph generation model with adaptive sparsity for node anomaly detection in IoT networks

    [https://arxiv.org/abs/2402.17363](https://arxiv.org/abs/2402.17363)

    CGGM是一种新颖的图生成模型，通过自适应稀疏性生成邻接矩阵，解决了物联网网络中节点异常检测中节点类别不平衡的问题

    

    动态图被广泛用于检测物联网中节点的异常行为。生成模型通常用于解决动态图中节点类别不平衡的问题。然而，它面临的约束包括邻接关系的单调性，为节点构建多维特征的困难，以及缺乏端到端生成多类节点的方法。本文提出了一种名为CGGM的新颖图生成模型，专门设计用于生成少数类别中更多节点。通过自适应稀疏性生成邻接矩阵的机制增强了其结构的灵活性。特征生成模块名为多维特征生成器（MFG），可生成包括拓扑信息在内的节点特征。标签被转换为嵌入向量，用作条件。

    arXiv:2402.17363v1 Announce Type: cross  Abstract: Dynamic graphs are extensively employed for detecting anomalous behavior in nodes within the Internet of Things (IoT). Generative models are often used to address the issue of imbalanced node categories in dynamic graphs. Nevertheless, the constraints it faces include the monotonicity of adjacency relationships, the difficulty in constructing multi-dimensional features for nodes, and the lack of a method for end-to-end generation of multiple categories of nodes. This paper presents a novel graph generation model, called CGGM, designed specifically to generate a larger number of nodes belonging to the minority class. The mechanism for generating an adjacency matrix, through adaptive sparsity, enhances flexibility in its structure. The feature generation module, called multidimensional features generator (MFG) to generate node features along with topological information. Labels are transformed into embedding vectors, serving as condition
    
[^2]: 将其推向展示极限：多模态视觉触觉模仿学习与力匹配

    Push it to the Demonstrated Limit: Multimodal Visuotactile Imitation Learning with Force Matching. (arXiv:2311.01248v1 [cs.RO])

    [http://arxiv.org/abs/2311.01248](http://arxiv.org/abs/2311.01248)

    本研究利用视觉触觉传感器和模仿学习相结合，通过配对优化触觉力量曲线和简化传感器应用，对接触丰富的操作任务进行了研究。

    

    光学触觉传感器已经成为机器人操作过程中获取密集接触信息的有效手段。最近引入的“透视你的皮肤”（STS）型传感器具有视觉和触觉模式，通过利用半透明表面和可控照明实现。本文研究了视觉触觉传感与模仿学习在富有接触的操作任务中的好处。首先，我们使用触觉力测量和一种新的算法，在运动示范中产生更好匹配人体示范者的力曲线。其次，我们添加了视觉/触觉STS模式切换作为控制策略输出，简化传感器的应用。最后，我们研究了多种观察配置，比较和对比了视觉/触觉数据（包括模式切换和不切换）与手腕挂载的眼在手摄像机的视觉数据的价值。我们在一个广泛的实验系列上进行实验。

    Optical tactile sensors have emerged as an effective means to acquire dense contact information during robotic manipulation. A recently-introduced `see-through-your-skin' (STS) variant of this type of sensor has both visual and tactile modes, enabled by leveraging a semi-transparent surface and controllable lighting. In this work, we investigate the benefits of pairing visuotactile sensing with imitation learning for contact-rich manipulation tasks. First, we use tactile force measurements and a novel algorithm during kinesthetic teaching to yield a force profile that better matches that of the human demonstrator. Second, we add visual/tactile STS mode switching as a control policy output, simplifying the application of the sensor. Finally, we study multiple observation configurations to compare and contrast the value of visual/tactile data (both with and without mode switching) with visual data from a wrist-mounted eye-in-hand camera. We perform an extensive series of experiments on a
    

