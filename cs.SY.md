# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Powerformer: A Section-adaptive Transformer for Power Flow Adjustment.](http://arxiv.org/abs/2401.02771) | Powerformer是一种适应不同传输区段的变压器架构，用于学习稳健电力系统状态表示。它通过开发专用的区段自适应注意机制，并引入图神经网络传播和多因素注意机制来提供更加稳健的状态表示。在三个不同的电力系统场景上进行了广泛评估。 |

# 详细

[^1]: Powerformer：适应不同传输区段的变压器架构用于电力流调整

    Powerformer: A Section-adaptive Transformer for Power Flow Adjustment. (arXiv:2401.02771v1 [cs.LG])

    [http://arxiv.org/abs/2401.02771](http://arxiv.org/abs/2401.02771)

    Powerformer是一种适应不同传输区段的变压器架构，用于学习稳健电力系统状态表示。它通过开发专用的区段自适应注意机制，并引入图神经网络传播和多因素注意机制来提供更加稳健的状态表示。在三个不同的电力系统场景上进行了广泛评估。

    

    本文提出了一种专为学习稳健电力系统状态表示而量身定制的变压器架构，旨在优化跨不同传输区段的电力调度以进行电力流调整。具体而言，我们的提出的方法名为Powerformer，开发了一种专用的区段自适应注意机制，与传统变压器中使用的自注意分离开来。该机制有效地将电力系统状态与传输区段信息整合在一起，有助于开发稳健的状态表示。此外，通过考虑电力系统的图拓扑和母线节点的电气属性，我们引入了两种定制策略来进一步增强表达能力：图神经网络传播和多因素注意机制。我们在三个电力系统场景（包括IEEE 118节点系统、中国实际300节点系统和一个大型系统）上进行了广泛的评估。

    In this paper, we present a novel transformer architecture tailored for learning robust power system state representations, which strives to optimize power dispatch for the power flow adjustment across different transmission sections. Specifically, our proposed approach, named Powerformer, develops a dedicated section-adaptive attention mechanism, separating itself from the self-attention used in conventional transformers. This mechanism effectively integrates power system states with transmission section information, which facilitates the development of robust state representations. Furthermore, by considering the graph topology of power system and the electrical attributes of bus nodes, we introduce two customized strategies to further enhance the expressiveness: graph neural network propagation and multi-factor attention mechanism. Extensive evaluations are conducted on three power system scenarios, including the IEEE 118-bus system, a realistic 300-bus system in China, and a large-
    

