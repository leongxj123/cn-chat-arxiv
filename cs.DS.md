# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [inGRASS: Incremental Graph Spectral Sparsification via Low-Resistance-Diameter Decomposition](https://arxiv.org/abs/2402.16990) | inGRASS提出了一种用于大型无向图增量谱稀疏化的新算法，其具有高度可扩展性和并行友好性，关键创新在于低阻抗直径分解方案，能够高效识别关键边和检测多余边。 |

# 详细

[^1]: inGRASS: 通过低阻抗直径分解实现增量图谱稀疏化

    inGRASS: Incremental Graph Spectral Sparsification via Low-Resistance-Diameter Decomposition

    [https://arxiv.org/abs/2402.16990](https://arxiv.org/abs/2402.16990)

    inGRASS提出了一种用于大型无向图增量谱稀疏化的新算法，其具有高度可扩展性和并行友好性，关键创新在于低阻抗直径分解方案，能够高效识别关键边和检测多余边。

    

    这项工作介绍了inGRASS，这是一种旨在对大型无向图进行增量谱稀疏化的新算法。所提出的inGRASS算法具有高度可扩展性和并行友好性，设置阶段的时间复杂度几乎是线性的，并且能够在对具有N个节点的原始图进行增量更改时，以$O(\log N)$的时间更新谱稀疏器。在inGRASS的设置阶段中，一个关键组件是引入了多级阻抗嵌入框架，用于高效识别谱关键边并有效检测多余边，这是通过将初始稀疏器分解为许多节点群集并利用低阻抗直径分解（LRD）方案来实现的。inGRASS的更新阶段利用低维节点嵌入向量，有效估计每个新添加边的重要性和唯一性。

    arXiv:2402.16990v1 Announce Type: cross  Abstract: This work presents inGRASS, a novel algorithm designed for incremental spectral sparsification of large undirected graphs. The proposed inGRASS algorithm is highly scalable and parallel-friendly, having a nearly-linear time complexity for the setup phase and the ability to update the spectral sparsifier in $O(\log N)$ time for each incremental change made to the original graph with $N$ nodes. A key component in the setup phase of inGRASS is a multilevel resistance embedding framework introduced for efficiently identifying spectrally-critical edges and effectively detecting redundant ones, which is achieved by decomposing the initial sparsifier into many node clusters with bounded effective-resistance diameters leveraging a low-resistance-diameter decomposition (LRD) scheme. The update phase of inGRASS exploits low-dimensional node embedding vectors for efficiently estimating the importance and uniqueness of each newly added edge. As de
    

