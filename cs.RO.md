# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Knolling Bot: Learning Robotic Object Arrangement from Tidy Demonstrations](https://arxiv.org/abs/2310.04566) | 本论文介绍了一种自监督学习框架，利用Transformer神经网络使机器人能够从整齐排列的示范中理解和复制整洁的概念，从而实现整理物品的功能。 |
| [^2] | [LPAC: Learnable Perception-Action-Communication Loops with Applications to Coverage Control.](http://arxiv.org/abs/2401.04855) | 提出了一种可学习的感知-行动-通信(LPAC)架构，使用卷积神经网络处理环境感知，图神经网络实现机器人之间的信息交流，浅层多层感知机计算机器人的动作。使用集中式显微算法训练模型，实现机器人群体的协作。 |

# 详细

[^1]: Knolling Bot: 从整洁的示范中学习机器人对象排列

    Knolling Bot: Learning Robotic Object Arrangement from Tidy Demonstrations

    [https://arxiv.org/abs/2310.04566](https://arxiv.org/abs/2310.04566)

    本论文介绍了一种自监督学习框架，利用Transformer神经网络使机器人能够从整齐排列的示范中理解和复制整洁的概念，从而实现整理物品的功能。

    

    地址：arXiv:2310.04566v2  公告类型：replace-cross  摘要：解决家庭空间中散乱物品的整理挑战受到整洁性的多样性和主观性的复杂性影响。正如人类语言的复杂性允许同一理念的多种表达一样，家庭整洁偏好和组织模式变化广泛，因此预设物体位置将限制对新物体和环境的适应性。受自然语言处理（NLP）的进展启发，本文引入一种自监督学习框架，使机器人能够从整洁布局的示范中理解和复制整洁的概念，类似于使用会话数据集训练大语言模型（LLM）。我们利用一个Transformer神经网络来预测后续物体的摆放位置。我们展示了一个“整理”系统，利用机械臂和RGB相机在桌子上组织不同大小和数量的物品。

    arXiv:2310.04566v2 Announce Type: replace-cross  Abstract: Addressing the challenge of organizing scattered items in domestic spaces is complicated by the diversity and subjective nature of tidiness. Just as the complexity of human language allows for multiple expressions of the same idea, household tidiness preferences and organizational patterns vary widely, so presetting object locations would limit the adaptability to new objects and environments. Inspired by advancements in natural language processing (NLP), this paper introduces a self-supervised learning framework that allows robots to understand and replicate the concept of tidiness from demonstrations of well-organized layouts, akin to using conversational datasets to train Large Language Models(LLM). We leverage a transformer neural network to predict the placement of subsequent objects. We demonstrate a ``knolling'' system with a robotic arm and an RGB camera to organize items of varying sizes and quantities on a table. Our 
    
[^2]: LPAC: 可学习的感知-行动-通信循环及其在覆盖控制中的应用

    LPAC: Learnable Perception-Action-Communication Loops with Applications to Coverage Control. (arXiv:2401.04855v1 [cs.RO])

    [http://arxiv.org/abs/2401.04855](http://arxiv.org/abs/2401.04855)

    提出了一种可学习的感知-行动-通信(LPAC)架构，使用卷积神经网络处理环境感知，图神经网络实现机器人之间的信息交流，浅层多层感知机计算机器人的动作。使用集中式显微算法训练模型，实现机器人群体的协作。

    

    覆盖控制是指导机器人群体协同监测未知的感兴趣特征或现象的问题。在有限的通信和感知能力的分散设置中，这个问题具有挑战性。本文提出了一种可学习的感知-行动-通信(LPAC)架构来解决覆盖控制问题。在该解决方案中，卷积神经网络(CNN)处理了环境的局部感知；图神经网络(GNN)实现了邻近机器人之间的相关信息通信；最后，浅层多层感知机(MLP)计算机器人的动作。通信模块中的GNN通过计算应该与邻居通信哪些信息以及如何利用接收到的信息采取适当的行动来实现机器人群体的协作。我们使用一个知晓整个环境的集中式显微算法来进行模型的训练。

    Coverage control is the problem of navigating a robot swarm to collaboratively monitor features or a phenomenon of interest not known a priori. The problem is challenging in decentralized settings with robots that have limited communication and sensing capabilities. This paper proposes a learnable Perception-Action-Communication (LPAC) architecture for the coverage control problem. In the proposed solution, a convolution neural network (CNN) processes localized perception of the environment; a graph neural network (GNN) enables communication of relevant information between neighboring robots; finally, a shallow multi-layer perceptron (MLP) computes robot actions. The GNN in the communication module enables collaboration in the robot swarm by computing what information to communicate with neighbors and how to use received information to take appropriate actions. We train models using imitation learning with a centralized clairvoyant algorithm that is aware of the entire environment. Eva
    

