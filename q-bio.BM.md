# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AlphaFold Meets Flow Matching for Generating Protein Ensembles](https://arxiv.org/abs/2402.04845) | 本研究开发了一种基于流动匹配的生成建模方法，称为AlphaFlow和ESMFlow，用于学习和采样蛋白质的构象空间。与AlphaFold相比，该方法在精度和多样性方面提供了更优的组合，在训练和评估时能准确捕捉到构象灵活性和高阶组合可观测性。同时，该方法可以将静态PDB结构多样化到特定的平衡性质，具有较快的收敛速度。 |
| [^2] | [From Static to Dynamic Structures: Improving Binding Affinity Prediction with a Graph-Based Deep Learning Model.](http://arxiv.org/abs/2208.10230) | 本文开发了一种名为 Dynaformer 的基于图的深度学习模型，利用分子动力学（MD）模拟中的蛋白质-配体相互作用几何特征来准确预测结合亲和力，并在CAS-2016基准数据集上展现了最先进的评分和排名能力。 |

# 详细

[^1]: AlphaFold遇到Flow Matching生成蛋白质集合

    AlphaFold Meets Flow Matching for Generating Protein Ensembles

    [https://arxiv.org/abs/2402.04845](https://arxiv.org/abs/2402.04845)

    本研究开发了一种基于流动匹配的生成建模方法，称为AlphaFlow和ESMFlow，用于学习和采样蛋白质的构象空间。与AlphaFold相比，该方法在精度和多样性方面提供了更优的组合，在训练和评估时能准确捕捉到构象灵活性和高阶组合可观测性。同时，该方法可以将静态PDB结构多样化到特定的平衡性质，具有较快的收敛速度。

    

    蛋白质的生物功能往往依赖于动态结构集合。本研究中，我们开发了一种基于流动匹配的生成建模方法，用于学习和采样蛋白质的构象空间。我们重新利用高精度的单态预测器，如AlphaFold和ESMFold，并在自定义流匹配框架下对其进行微调，以获得基于序列条件的蛋白质结构生成模型，称为AlphaFlow和ESMFlow。在PDB上进行训练和评估时，我们的方法相比于AlphaFold和MSA子采样提供了更高的精度和多样性的组合。当进一步训练所有原子MD的组合时，我们的方法可以准确地捕捉到未见蛋白质的构象灵活性、位置分布和高阶组合可观测性。此外，我们的方法可以通过更快的时钟收敛速度将静态PDB结构多样化到特定的平衡性质，比复制的MD轨迹更具潜力。

    The biological functions of proteins often depend on dynamic structural ensembles. In this work, we develop a flow-based generative modeling approach for learning and sampling the conformational landscapes of proteins. We repurpose highly accurate single-state predictors such as AlphaFold and ESMFold and fine-tune them under a custom flow matching framework to obtain sequence-conditoned generative models of protein structure called AlphaFlow and ESMFlow. When trained and evaluated on the PDB, our method provides a superior combination of precision and diversity compared to AlphaFold with MSA subsampling. When further trained on ensembles from all-atom MD, our method accurately captures conformational flexibility, positional distributions, and higher-order ensemble observables for unseen proteins. Moreover, our method can diversify a static PDB structure with faster wall-clock convergence to certain equilibrium properties than replicate MD trajectories, demonstrating its potential as a 
    
[^2]: 从静态到动态的结构：利用基于图的深度学习模型提高结合亲和性预测

    From Static to Dynamic Structures: Improving Binding Affinity Prediction with a Graph-Based Deep Learning Model. (arXiv:2208.10230v3 [q-bio.BM] UPDATED)

    [http://arxiv.org/abs/2208.10230](http://arxiv.org/abs/2208.10230)

    本文开发了一种名为 Dynaformer 的基于图的深度学习模型，利用分子动力学（MD）模拟中的蛋白质-配体相互作用几何特征来准确预测结合亲和力，并在CAS-2016基准数据集上展现了最先进的评分和排名能力。

    

    准确预测蛋白质配体结合亲和力是结构基础药物设计中的重要挑战，虽然数据驱动方法在亲和力预测中有所进展，但其准确性仍然受限，部分原因是因为它们只利用静态晶体结构，而实际的结合亲和力通常由蛋白质和配体之间的热力学集合描述。逼近这样的热力学集合的有效方法是使用分子动力学（MD）模拟。本文整理了一个包含3,218个不同蛋白质-配体复合物的MD数据集，并进一步开发了一种名为Dynaformer的基于图的深度学习模型。 Dynaformer能够通过学习从MD轨迹中蛋白质-配体相互作用的几何特征来准确预测结合亲和力。体外实验表明，我们的模型在CASF-2016基准数据集上展现了最先进的评分和排名能力。

    Accurate prediction of the protein-ligand binding affinities is an essential challenge in the structure-based drug design. Despite recent advance in data-driven methods in affinity prediction, their accuracy is still limited, partially because they only take advantage of static crystal structures while the actual binding affinities are generally depicted by the thermodynamic ensembles between proteins and ligands. One effective way to approximate such a thermodynamic ensemble is to use molecular dynamics (MD) simulation. Here, we curated an MD dataset containing 3,218 different protein-ligand complexes, and further developed Dynaformer, which is a graph-based deep learning model. Dynaformer was able to accurately predict the binding affinities by learning the geometric characteristics of the protein-ligand interactions from the MD trajectories. In silico experiments demonstrated that our model exhibits state-of-the-art scoring and ranking power on the CASF-2016 benchmark dataset, outpe
    

