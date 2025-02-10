# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Group Symmetric Stochastic Differential Equation Model for Molecule Multi-modal Pretraining.](http://arxiv.org/abs/2305.18407) | MoleculeSDE是用于分子多模态预训练的群对称随机微分方程模型，通过在输入空间中直接生成3D几何与2D拓扑之间的转换，它能够更有效地保存分子结构信息。 |

# 详细

[^1]: 一种用于分子多模态预训练的群对称随机微分方程模型。

    A Group Symmetric Stochastic Differential Equation Model for Molecule Multi-modal Pretraining. (arXiv:2305.18407v1 [cs.LG])

    [http://arxiv.org/abs/2305.18407](http://arxiv.org/abs/2305.18407)

    MoleculeSDE是用于分子多模态预训练的群对称随机微分方程模型，通过在输入空间中直接生成3D几何与2D拓扑之间的转换，它能够更有效地保存分子结构信息。

    

    分子预训练已经成为提高基于 AI 的药物发现性能的主流方法。然而，大部分现有的方法只关注单一的模态。最近的研究表明，最大化两种模态之间的互信息（MI）可以增强分子表示能力。而现有的分子多模态预训练方法基于从拓扑和几何编码的表示空间来估计 MI，因此丢失了分子的关键结构信息。为解决这一问题，我们提出了 MoleculeSDE。MoleculeSDE利用群对称（如 SE（3）-等变和反射-反对称）随机微分方程模型在输入空间中直接生成 3D 几何形状与 2D 拓扑之间的转换。它不仅获得更紧的MI界限，而且还能够有效地保存分子结构信息。

    Molecule pretraining has quickly become the go-to schema to boost the performance of AI-based drug discovery. Naturally, molecules can be represented as 2D topological graphs or 3D geometric point clouds. Although most existing pertaining methods focus on merely the single modality, recent research has shown that maximizing the mutual information (MI) between such two modalities enhances the molecule representation ability. Meanwhile, existing molecule multi-modal pretraining approaches approximate MI based on the representation space encoded from the topology and geometry, thus resulting in the loss of critical structural information of molecules. To address this issue, we propose MoleculeSDE. MoleculeSDE leverages group symmetric (e.g., SE(3)-equivariant and reflection-antisymmetric) stochastic differential equation models to generate the 3D geometries from 2D topologies, and vice versa, directly in the input space. It not only obtains tighter MI bound but also enables prosperous dow
    

