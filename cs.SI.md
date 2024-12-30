# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generative Modeling of Graphs via Joint Diffusion of Node and Edge Attributes](https://arxiv.org/abs/2402.04046) | 通过联合扩散节点和边属性，我们提出了一个新的图形生成模型，考虑了所有图组件，并通过注意模块和相互依赖的节点、边和邻接信息实现了更好的效果。 |

# 详细

[^1]: 通过节点和边属性的联合扩散，实现图形的生成建模

    Generative Modeling of Graphs via Joint Diffusion of Node and Edge Attributes

    [https://arxiv.org/abs/2402.04046](https://arxiv.org/abs/2402.04046)

    通过联合扩散节点和边属性，我们提出了一个新的图形生成模型，考虑了所有图组件，并通过注意模块和相互依赖的节点、边和邻接信息实现了更好的效果。

    

    图生成是各种工程和科学学科的基础。然而，现有的方法往往忽视了边属性的生成。然而，我们确定了一些关键应用中边属性的重要性，这使得先前的方法在这些情境中可能不适用。此外，虽然存在一些简单的适应方法，但经验调查显示它们的效果有限，因为它们没有很好地模拟图组件之间的相互作用。为了解决这个问题，我们提出了一个节点和边的联合评分模型，用于图形生成，考虑了所有图组件。我们的方法具有两个关键创新点：(i) 将节点和边属性结合在一个注意模块中，基于这两个因素生成样本；(ii) 在图形扩散过程中，节点、边和邻接信息相互依赖。我们在涉及实际和合成数据集的具有挑战性的基准测试中评估了我们的方法，其中包含边特征。

    Graph generation is integral to various engineering and scientific disciplines. Nevertheless, existing methodologies tend to overlook the generation of edge attributes. However, we identify critical applications where edge attributes are essential, making prior methods potentially unsuitable in such contexts. Moreover, while trivial adaptations are available, empirical investigations reveal their limited efficacy as they do not properly model the interplay among graph components. To address this, we propose a joint score-based model of nodes and edges for graph generation that considers all graph components. Our approach offers two key novelties: (i) node and edge attributes are combined in an attention module that generates samples based on the two ingredients; and (ii) node, edge and adjacency information are mutually dependent during the graph diffusion process. We evaluate our method on challenging benchmarks involving real-world and synthetic datasets in which edge features are cr
    

