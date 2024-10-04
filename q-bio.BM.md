# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [3M-Diffusion: Latent Multi-Modal Diffusion for Text-Guided Generation of Molecular Graphs](https://arxiv.org/abs/2403.07179) | 提出了3M-Diffusion，一种新颖的多模态分子图生成方法，可以生成具有所需属性的多样化、理想情况下是新颖的分子。 |
| [^2] | [Inverse Molecular Design with Multi-Conditional Diffusion Guidance.](http://arxiv.org/abs/2401.13858) | 借助多条件扩散引导的逆分子设计模型在材料和药物发现方面具有巨大潜力。通过引入Transformer-based去噪模型和图依赖的扩散过程，该模型能够在多个条件约束下准确地生成聚合物和小分子。 |
| [^3] | [AtomSurf : Surface Representation for Learning on Protein Structures.](http://arxiv.org/abs/2309.16519) | 本文研究了将蛋白质作为3D网格的表面表示，并提出了一种结合图表面的协同方法，既有竞争优势，又有实际应用潜力。 |

# 详细

[^1]: 3M-Diffusion：用于文本引导生成分子图的潜在多模态扩散

    3M-Diffusion: Latent Multi-Modal Diffusion for Text-Guided Generation of Molecular Graphs

    [https://arxiv.org/abs/2403.07179](https://arxiv.org/abs/2403.07179)

    提出了3M-Diffusion，一种新颖的多模态分子图生成方法，可以生成具有所需属性的多样化、理想情况下是新颖的分子。

    

    生成具有所需属性的分子是一项关键任务，在药物发现和材料设计中具有广泛应用。受到大型语言模型的最新进展的启发，越来越多的人对使用分子的自然语言描述来生成具有所需属性的分子产生了兴趣。大多数现有方法侧重于生成与文本描述精确匹配的分子。然而，实际应用需要能够生成具有所需属性的多样化，理想情况下是新颖的分子的方法。我们提出了一种新颖的多模态分子图生成方法3M-Diffusion，以解决这一挑战。

    arXiv:2403.07179v1 Announce Type: cross  Abstract: Generating molecules with desired properties is a critical task with broad applications in drug discovery and materials design. Inspired by recent advances in large language models, there is a growing interest in using natural language descriptions of molecules to generate molecules with the desired properties. Most existing methods focus on generating molecules that precisely match the text description. However, practical applications call for methods that generate diverse, and ideally novel, molecules with the desired properties. We propose 3M-Diffusion, a novel multi-modal molecular graph generation method, to address this challenge. 3M-Diffusion first encodes molecular graphs into a graph latent space aligned with text descriptions. It then reconstructs the molecular structure and atomic attributes based on the given text descriptions using the molecule decoder. It then learns a probabilistic mapping from the text space to the late
    
[^2]: 借助多条件扩散引导的逆分子设计

    Inverse Molecular Design with Multi-Conditional Diffusion Guidance. (arXiv:2401.13858v1 [cs.LG])

    [http://arxiv.org/abs/2401.13858](http://arxiv.org/abs/2401.13858)

    借助多条件扩散引导的逆分子设计模型在材料和药物发现方面具有巨大潜力。通过引入Transformer-based去噪模型和图依赖的扩散过程，该模型能够在多个条件约束下准确地生成聚合物和小分子。

    

    借助扩散模型进行逆分子设计在材料和药物发现方面具有巨大潜力。虽然在无条件分子生成方面取得了成功，但将合成评分和气体渗透性等多个属性作为条件约束集成到扩散模型中仍未被探索。我们引入了多条件扩散引导。所提出的基于Transformer的去噪模型具有一个条件编码器，该编码器学习了数值和分类条件的表示。组成结构编码器-解码器的去噪模型在条件表示下进行去噪训练。扩散过程变得依赖于图来准确估计分子中与图相关的噪声，而不像以前的模型仅关注原子或键的边缘分布。我们广泛验证了我们的模型在多条件聚合物和小分子生成方面的优越性。结果显示我们在分布度量方面的优势。

    Inverse molecular design with diffusion models holds great potential for advancements in material and drug discovery. Despite success in unconditional molecule generation, integrating multiple properties such as synthetic score and gas permeability as condition constraints into diffusion models remains unexplored. We introduce multi-conditional diffusion guidance. The proposed Transformer-based denoising model has a condition encoder that learns the representations of numerical and categorical conditions. The denoising model, consisting of a structure encoder-decoder, is trained for denoising under the representation of conditions. The diffusion process becomes graph-dependent to accurately estimate graph-related noise in molecules, unlike the previous models that focus solely on the marginal distributions of atoms or bonds. We extensively validate our model for multi-conditional polymer and small molecule generation. Results demonstrate our superiority across metrics from distribution
    
[^3]: AtomSurf：蛋白质结构上的学习的表面表示

    AtomSurf : Surface Representation for Learning on Protein Structures. (arXiv:2309.16519v1 [cs.LG])

    [http://arxiv.org/abs/2309.16519](http://arxiv.org/abs/2309.16519)

    本文研究了将蛋白质作为3D网格的表面表示，并提出了一种结合图表面的协同方法，既有竞争优势，又有实际应用潜力。

    

    近期Cryo-EM和蛋白质结构预测算法的进展使得大规模蛋白质结构可获得，为基于机器学习的功能注释铺平了道路。几何深度学习领域关注创建适用于几何数据的方法。从蛋白质结构中学习的一个重要方面是将这些结构表示为几何对象（如网格、图或表面）并应用适合这种表示形式的学习方法。给定方法的性能将取决于表示和相应的学习方法。在本文中，我们研究将蛋白质表示为$\textit{3D mesh surfaces}$并将其纳入已建立的表示基准中。我们的第一个发现是，尽管有着有希望的初步结果，但仅单独表面表示似乎无法与3D网格竞争。在此基础上，我们提出了一种协同方法，将表面表示与图表面结合起来。

    Recent advancements in Cryo-EM and protein structure prediction algorithms have made large-scale protein structures accessible, paving the way for machine learning-based functional annotations.The field of geometric deep learning focuses on creating methods working on geometric data. An essential aspect of learning from protein structures is representing these structures as a geometric object (be it a grid, graph, or surface) and applying a learning method tailored to this representation. The performance of a given approach will then depend on both the representation and its corresponding learning method.  In this paper, we investigate representing proteins as $\textit{3D mesh surfaces}$ and incorporate them into an established representation benchmark. Our first finding is that despite promising preliminary results, the surface representation alone does not seem competitive with 3D grids. Building on this, we introduce a synergistic approach, combining surface representations with gra
    

