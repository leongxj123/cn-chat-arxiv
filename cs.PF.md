# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [InkStream: Real-time GNN Inference on Streaming Graphs via Incremental Update.](http://arxiv.org/abs/2309.11071) | InkStream是一种在流式图上进行实时推理的方法，通过增量更新节点嵌入来解决传统图神经网络在流式图上的挑战。 |

# 详细

[^1]: InkStream: 通过增量更新在流式图上进行实时的图神经网络推理

    InkStream: Real-time GNN Inference on Streaming Graphs via Incremental Update. (arXiv:2309.11071v1 [cs.LG])

    [http://arxiv.org/abs/2309.11071](http://arxiv.org/abs/2309.11071)

    InkStream是一种在流式图上进行实时推理的方法，通过增量更新节点嵌入来解决传统图神经网络在流式图上的挑战。

    

    传统的图神经网络推理方法适用于静态图，而对于随时间演变的流式图则不合适。流式图的动态性需要进行持续的更新，对GPU加速提出了独特的挑战。我们基于两个关键观点来解决这些挑战：（1）在k-hop邻域内，当模型使用最小或最大聚合函数时，只有一小部分节点受到修改边的影响；（2）当模型权重保持静态而图结构发生变化时，节点嵌入可以通过仅计算邻域的受影响部分来逐步演化。基于这些观点，我们提出了一种新颖的方法InkStream，旨在实现实时推理，最小化内存访问和计算，并确保与传统方法相同的输出。InkStream的操作原则是仅在必要时传播和获取数据。它使用基于事件的系统来控制。

    Classic Graph Neural Network (GNN) inference approaches, designed for static graphs, are ill-suited for streaming graphs that evolve with time. The dynamism intrinsic to streaming graphs necessitates constant updates, posing unique challenges to acceleration on GPU. We address these challenges based on two key insights: (1) Inside the $k$-hop neighborhood, a significant fraction of the nodes is not impacted by the modified edges when the model uses min or max as aggregation function; (2) When the model weights remain static while the graph structure changes, node embeddings can incrementally evolve over time by computing only the impacted part of the neighborhood. With these insights, we propose a novel method, InkStream, designed for real-time inference with minimal memory access and computation, while ensuring an identical output to conventional methods. InkStream operates on the principle of propagating and fetching data only when necessary. It uses an event-based system to control 
    

