# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Protein Conformation Generation via Force-Guided SE(3) Diffusion Models](https://arxiv.org/abs/2403.14088) | 本文提出了一种力引导SE(3)扩散模型ConfDiff，用于蛋白质构象生成，通过结合力引导网络与基于数据的分数模型，实现了对蛋白质构象的准确生成。 |
| [^2] | [Structure-based Drug Design with Equivariant Diffusion Models.](http://arxiv.org/abs/2210.13695) | 该论文提出了一个基于结构的药物设计方法，使用了等变扩散模型DiffSBDD来生成具有亲和力和特异性的新型药物配体。实验结果表明DiffSBDD在生成具有竞争性对接得分的多样化药物样配体方面具有高效和有效的性能。 |

# 详细

[^1]: 基于力引导SE(3)扩散模型的蛋白质构象生成

    Protein Conformation Generation via Force-Guided SE(3) Diffusion Models

    [https://arxiv.org/abs/2403.14088](https://arxiv.org/abs/2403.14088)

    本文提出了一种力引导SE(3)扩散模型ConfDiff，用于蛋白质构象生成，通过结合力引导网络与基于数据的分数模型，实现了对蛋白质构象的准确生成。

    

    蛋白质的构象景观对于理解复杂生物过程中的功能至关重要。传统基于物理的计算方法，如分子动力学（MD）模拟，存在稀有事件采样和长时间平衡问题，限制了它们在一般蛋白质系统中的应用。最近，深度生成建模技术，特别是扩散模型，已被应用于生成新颖的蛋白质构象。然而，现有的基于分数的扩散方法无法很好地结合重要的物理先验知识来指导生成过程，导致采样蛋白质构象与平衡分布之间存在较大偏差。本文提出了一种用于蛋白质构象生成的力引导SE(3)扩散模型ConfDiff，以克服这些限制。通过将力引导网络与一系列基于数据的分数模型相结合，ConfDiff能够实现对蛋白质构象的准确生成。

    arXiv:2403.14088v1 Announce Type: cross  Abstract: The conformational landscape of proteins is crucial to understanding their functionality in complex biological processes. Traditional physics-based computational methods, such as molecular dynamics (MD) simulations, suffer from rare event sampling and long equilibration time problems, hindering their applications in general protein systems. Recently, deep generative modeling techniques, especially diffusion models, have been employed to generate novel protein conformations. However, existing score-based diffusion methods cannot properly incorporate important physical prior knowledge to guide the generation process, causing large deviations in the sampled protein conformations from the equilibrium distribution. In this paper, to overcome these limitations, we propose a force-guided SE(3) diffusion model, ConfDiff, for protein conformation generation. By incorporating a force-guided network with a mixture of data-based score models, Conf
    
[^2]: 基于结构的药物设计与等变扩散模型

    Structure-based Drug Design with Equivariant Diffusion Models. (arXiv:2210.13695v2 [q-bio.BM] UPDATED)

    [http://arxiv.org/abs/2210.13695](http://arxiv.org/abs/2210.13695)

    该论文提出了一个基于结构的药物设计方法，使用了等变扩散模型DiffSBDD来生成具有亲和力和特异性的新型药物配体。实验结果表明DiffSBDD在生成具有竞争性对接得分的多样化药物样配体方面具有高效和有效的性能。

    

    基于结构的药物设计（SBDD）旨在设计与预定的蛋白靶点具有高亲和力和特异性的小分子配体。本文将SBDD表述为一个三维条件生成问题，并提出了DiffSBDD，这是一个SE(3)-等变的三维条件扩散模型，可以在蛋白口袋的条件下生成新型配体。全面的基于计算机模拟的实验证明了DiffSBDD在生成具有具有竞争性对接得分的新颖和多样的药物样配体方面的效率和有效性。我们进一步探索了扩散框架在药物设计活动中更广泛任务的灵活性，例如即插即用的性质优化和从局部分子设计带有修补的任务。

    Structure-based drug design (SBDD) aims to design small-molecule ligands that bind with high affinity and specificity to pre-determined protein targets. In this paper, we formulate SBDD as a 3D-conditional generation problem and present DiffSBDD, an SE(3)-equivariant 3D-conditional diffusion model that generates novel ligands conditioned on protein pockets. Comprehensive in silico experiments demonstrate the efficiency and effectiveness of DiffSBDD in generating novel and diverse drug-like ligands with competitive docking scores. We further explore the flexibility of the diffusion framework for a broader range of tasks in drug design campaigns, such as off-the-shelf property optimization and partial molecular design with inpainting.
    

