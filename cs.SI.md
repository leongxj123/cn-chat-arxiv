# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Disentangled Condensation for Large-scale Graphs.](http://arxiv.org/abs/2401.12231) | 本文提出了用于大规模图的解缠结凝聚方法DisCo，通过节点和边的凝聚模块实现了对大规模图的高效缩凝，提高了可扩展性和压缩图的保真度。 |

# 详细

[^1]: 大规模图的解缠结凝聚

    Disentangled Condensation for Large-scale Graphs. (arXiv:2401.12231v1 [cs.SI])

    [http://arxiv.org/abs/2401.12231](http://arxiv.org/abs/2401.12231)

    本文提出了用于大规模图的解缠结凝聚方法DisCo，通过节点和边的凝聚模块实现了对大规模图的高效缩凝，提高了可扩展性和压缩图的保真度。

    

    图解缠结已经成为一种有趣的技术，为大规模图提供了一种更紧凑但信息丰富的小图，以节省大规模图学习的昂贵成本。尽管取得了有前途的结果，但先前的图解缠结方法常常采用纠缠的缩凝策略，同时涉及节点和边的缩凝，导致大量的GPU内存需求。这种纠缠的策略极大地阻碍了图解缠结的可扩展性，削弱了它对极大规模图的缩凝和高保真度压缩图的能力。因此，本文提出了用于大规模图的解缠结凝聚，简称为DisCo，以提供可扩展的图解缠结，适用于不同规模的图。DisCo的核心是两个互补的组件，即节点和边的凝聚模块，在解缠的方式下实现节点和边的凝聚。

    Graph condensation has emerged as an intriguing technique to provide Graph Neural Networks for large-scale graphs with a more compact yet informative small graph to save the expensive costs of large-scale graph learning. Despite the promising results achieved, previous graph condensation methods often employ an entangled condensation strategy that involves condensing nodes and edges simultaneously, leading to substantial GPU memory demands. This entangled strategy has considerably impeded the scalability of graph condensation, impairing its capability to condense extremely large-scale graphs and produce condensed graphs with high fidelity. Therefore, this paper presents Disentangled Condensation for large-scale graphs, abbreviated as DisCo, to provide scalable graph condensation for graphs of varying sizes. At the heart of DisCo are two complementary components, namely node and edge condensation modules, that realize the condensation of nodes and edges in a disentangled manner. In the 
    

