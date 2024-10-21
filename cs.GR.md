# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Polyhedral Complex Derivation from Piecewise Trilinear Networks](https://arxiv.org/abs/2402.10403) | 本文以三线性插值方法作为位置编码，提出了理论见解和分析网格提取方法，将高维曲面转换为平面，并引入了一种近似交点的方法，拓展了更广泛的应用。 |
| [^2] | [Efficient Anatomical labeling of Pulmonary Tree Structures via Implicit Point-Graph Networks.](http://arxiv.org/abs/2309.17329) | 本文介绍了一种通过隐式点图网络高效解剖标记肺部树状结构的方法，提供了SOTA准确度和可用的表面，同时还提供了一个用于评估该方法的数据集。 |

# 详细

[^1]: 从分段三线性网络中导出多面体复合体

    Polyhedral Complex Derivation from Piecewise Trilinear Networks

    [https://arxiv.org/abs/2402.10403](https://arxiv.org/abs/2402.10403)

    本文以三线性插值方法作为位置编码，提出了理论见解和分析网格提取方法，将高维曲面转换为平面，并引入了一种近似交点的方法，拓展了更广泛的应用。

    

    最近关于深度神经网络可视化的进展揭示了它们结构的见解，并且可以从连续分段仿射（CPWA）函数中提取网格。与此同时，神经表面表示学习的发展包括非线性位置编码，解决了诸如谱偏差之类的问题；然而，这在应用基于CPWA函数的网格提取技术方面带来了挑战。我们聚焦于三线性插值方法作为位置编码，提供了理论见解和分析的网格提取，展示了在奇拿尔约束下将高维曲面转换为三线性区域内的平面的过程。此外，我们引入了一种方法来近似三个高维曲面之间的交点，从而扩展了更广泛的应用。通过汉明距离和效率以及角距离来经验性地验证正确性和简洁性，同时检查了t之间的相关性

    arXiv:2402.10403v1 Announce Type: cross  Abstract: Recent advancements in visualizing deep neural networks provide insights into their structures and mesh extraction from Continuous Piecewise Affine (CPWA) functions. Meanwhile, developments in neural surface representation learning incorporate non-linear positional encoding, addressing issues like spectral bias; however, this poses challenges in applying mesh extraction techniques based on CPWA functions. Focusing on trilinear interpolating methods as positional encoding, we present theoretical insights and an analytical mesh extraction, showing the transformation of hypersurfaces to flat planes within the trilinear region under the eikonal constraint. Moreover, we introduce a method for approximating intersecting points among three hypersurfaces contributing to broader applications. We empirically validate correctness and parsimony through chamfer distance and efficiency, and angular distance, while examining the correlation between t
    
[^2]: 通过隐式点图网络高效解剖标记肺部树状结构

    Efficient Anatomical labeling of Pulmonary Tree Structures via Implicit Point-Graph Networks. (arXiv:2309.17329v1 [cs.CV])

    [http://arxiv.org/abs/2309.17329](http://arxiv.org/abs/2309.17329)

    本文介绍了一种通过隐式点图网络高效解剖标记肺部树状结构的方法，提供了SOTA准确度和可用的表面，同时还提供了一个用于评估该方法的数据集。

    

    肺部疾病在全球范围内是导致死亡的主要原因之一。治愈肺部疾病需要更好地理解肺部系统内的许多复杂的3D树状结构，如气道、动脉和静脉。在理论上，它们可以通过高分辨率图像堆栈进行建模。然而，基于密集体素网格的标准CNN方法代价过高。为了解决这个问题，我们引入了一种基于点的方法，保留了树骨架的图连通性，并结合了隐式表面表示。它以较低的计算成本提供了SOTA准确度，生成的模型具有可用的表面。由于公开可访问的数据稀缺，我们还整理了一套广泛的数据集来评估我们的方法，并将其公开。

    Pulmonary diseases rank prominently among the principal causes of death worldwide. Curing them will require, among other things, a better understanding of the many complex 3D tree-shaped structures within the pulmonary system, such as airways, arteries, and veins. In theory, they can be modeled using high-resolution image stacks. Unfortunately, standard CNN approaches operating on dense voxel grids are prohibitively expensive. To remedy this, we introduce a point-based approach that preserves graph connectivity of tree skeleton and incorporates an implicit surface representation. It delivers SOTA accuracy at a low computational cost and the resulting models have usable surfaces. Due to the scarcity of publicly accessible data, we have also curated an extensive dataset to evaluate our approach and will make it public.
    

