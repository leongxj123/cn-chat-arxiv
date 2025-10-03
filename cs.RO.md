# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LPAC: Learnable Perception-Action-Communication Loops with Applications to Coverage Control.](http://arxiv.org/abs/2401.04855) | 提出了一种可学习的感知-行动-通信(LPAC)架构，使用卷积神经网络处理环境感知，图神经网络实现机器人之间的信息交流，浅层多层感知机计算机器人的动作。使用集中式显微算法训练模型，实现机器人群体的协作。 |

# 详细

[^1]: LPAC: 可学习的感知-行动-通信循环及其在覆盖控制中的应用

    LPAC: Learnable Perception-Action-Communication Loops with Applications to Coverage Control. (arXiv:2401.04855v1 [cs.RO])

    [http://arxiv.org/abs/2401.04855](http://arxiv.org/abs/2401.04855)

    提出了一种可学习的感知-行动-通信(LPAC)架构，使用卷积神经网络处理环境感知，图神经网络实现机器人之间的信息交流，浅层多层感知机计算机器人的动作。使用集中式显微算法训练模型，实现机器人群体的协作。

    

    覆盖控制是指导机器人群体协同监测未知的感兴趣特征或现象的问题。在有限的通信和感知能力的分散设置中，这个问题具有挑战性。本文提出了一种可学习的感知-行动-通信(LPAC)架构来解决覆盖控制问题。在该解决方案中，卷积神经网络(CNN)处理了环境的局部感知；图神经网络(GNN)实现了邻近机器人之间的相关信息通信；最后，浅层多层感知机(MLP)计算机器人的动作。通信模块中的GNN通过计算应该与邻居通信哪些信息以及如何利用接收到的信息采取适当的行动来实现机器人群体的协作。我们使用一个知晓整个环境的集中式显微算法来进行模型的训练。

    Coverage control is the problem of navigating a robot swarm to collaboratively monitor features or a phenomenon of interest not known a priori. The problem is challenging in decentralized settings with robots that have limited communication and sensing capabilities. This paper proposes a learnable Perception-Action-Communication (LPAC) architecture for the coverage control problem. In the proposed solution, a convolution neural network (CNN) processes localized perception of the environment; a graph neural network (GNN) enables communication of relevant information between neighboring robots; finally, a shallow multi-layer perceptron (MLP) computes robot actions. The GNN in the communication module enables collaboration in the robot swarm by computing what information to communicate with neighbors and how to use received information to take appropriate actions. We train models using imitation learning with a centralized clairvoyant algorithm that is aware of the entire environment. Eva
    

