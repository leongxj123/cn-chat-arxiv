# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FedStruct: Federated Decoupled Learning over Interconnected Graphs](https://arxiv.org/abs/2402.19163) | FedStruct提出了一种新的框架，利用深层结构依赖关系在互联图上进行联合解耦学习，有效地维护隐私并捕捉节点间的依赖关系。 |

# 详细

[^1]: FedStruct：联合解耦学习在互联图上

    FedStruct: Federated Decoupled Learning over Interconnected Graphs

    [https://arxiv.org/abs/2402.19163](https://arxiv.org/abs/2402.19163)

    FedStruct提出了一种新的框架，利用深层结构依赖关系在互联图上进行联合解耦学习，有效地维护隐私并捕捉节点间的依赖关系。

    

    我们解决了分布在多个客户端上的图结构数据上的联合学习挑战。具体来说，我们关注互联子图的普遍情况，其中不同客户端之间的相互连接起着关键作用。我们提出了针对这种情况的一种新颖框架，名为FedStruct，它利用深层结构依赖关系。为了维护隐私，与现有方法不同，FedStruct消除了在客户端之间共享或生成敏感节点特征或嵌入的必要性。相反，它利用显式全局图结构信息来捕捉节点间的依赖关系。我们通过在六个数据集上进行的实验结果验证了FedStruct的有效性，展示了在各种情况下（包括不同数据分区方法、不同标签可用性以及客户个数的）接近于集中式方法的性能。

    arXiv:2402.19163v1 Announce Type: new  Abstract: We address the challenge of federated learning on graph-structured data distributed across multiple clients. Specifically, we focus on the prevalent scenario of interconnected subgraphs, where inter-connections between different clients play a critical role. We present a novel framework for this scenario, named FedStruct, that harnesses deep structural dependencies. To uphold privacy, unlike existing methods, FedStruct eliminates the necessity of sharing or generating sensitive node features or embeddings among clients. Instead, it leverages explicit global graph structure information to capture inter-node dependencies. We validate the effectiveness of FedStruct through experimental results conducted on six datasets for semi-supervised node classification, showcasing performance close to the centralized approach across various scenarios, including different data partitioning methods, varying levels of label availability, and number of cl
    

