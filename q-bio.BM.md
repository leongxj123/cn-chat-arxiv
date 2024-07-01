# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Impact of Domain Knowledge and Multi-Modality on Intelligent Molecular Property Prediction: A Systematic Survey](https://arxiv.org/abs/2402.07249) | 本文通过系统调查，发现整合领域知识可以提高分子性质预测的准确性，同时利用多模态数据融合可以产生更精确的结果。 |
| [^2] | [3D-Mol: A Novel Contrastive Learning Framework for Molecular Property Prediction with 3D Information.](http://arxiv.org/abs/2309.17366) | 3D-Mol是一种新颖的基于3D结构的分子建模方法，通过对比学习提高了分子性质预测准确性，并在多个基准数据集上超过了最先进的模型。 |
| [^3] | [Importance Weighted Expectation-Maximization for Protein Sequence Design.](http://arxiv.org/abs/2305.00386) | 本文提出了一种名为IsEM-Pro的方法，用于根据给定适应性标准生成蛋白质序列。在推理期间，从其潜在空间采样可以增加多样性，指导了探索高适应性区域。实验表明，相比先前最佳方法，IsEM-Pro的平均适应性得分至少高出55％，并生成了更多样化和新颖的蛋白质序列。 |

# 详细

[^1]: 领域知识和多模态对智能分子性质预测的影响：一项系统调查

    The Impact of Domain Knowledge and Multi-Modality on Intelligent Molecular Property Prediction: A Systematic Survey

    [https://arxiv.org/abs/2402.07249](https://arxiv.org/abs/2402.07249)

    本文通过系统调查，发现整合领域知识可以提高分子性质预测的准确性，同时利用多模态数据融合可以产生更精确的结果。

    

    准确预测分子性质对于药物开发尤其是虚拟筛选和化合物优化的进展至关重要。近年来引入了许多基于深度学习的方法，在增强分子性质预测（MPP）方面显示出显著潜力，特别是提高了准确性和对分子结构的洞察力。然而，有两个关键问题：领域知识的整合是否增强了分子性质预测的准确性，使用多模态数据融合是否比单一数据来源方法产生更精确的结果？为了探究这些问题，我们全面回顾和定量分析了基于各种基准的最新深度学习方法。我们发现，整合分子信息将分别提高MPP回归和分类任务的准确性，分别高达3.98％和1.72％。我们还发现，使用三维信息与一维和二维信息相结合会产生更好的结果。

    The precise prediction of molecular properties is essential for advancements in drug development, particularly in virtual screening and compound optimization. The recent introduction of numerous deep learning-based methods has shown remarkable potential in enhancing molecular property prediction (MPP), especially improving accuracy and insights into molecular structures. Yet, two critical questions arise: does the integration of domain knowledge augment the accuracy of molecular property prediction and does employing multi-modal data fusion yield more precise results than unique data source methods? To explore these matters, we comprehensively review and quantitatively analyze recent deep learning methods based on various benchmarks. We discover that integrating molecular information will improve both MPP regression and classification tasks by upto 3.98% and 1.72%, respectively. We also discover that the utilizing 3-dimensional information with 1-dimensional and 2-dimensional informati
    
[^2]: 3D-Mol: 一种新颖的基于对比学习的分子性质预测框架，利用了3D信息

    3D-Mol: A Novel Contrastive Learning Framework for Molecular Property Prediction with 3D Information. (arXiv:2309.17366v1 [q-bio.BM])

    [http://arxiv.org/abs/2309.17366](http://arxiv.org/abs/2309.17366)

    3D-Mol是一种新颖的基于3D结构的分子建模方法，通过对比学习提高了分子性质预测准确性，并在多个基准数据集上超过了最先进的模型。

    

    分子性质预测为药物候选物的早期筛选和优化提供了一种有效且高效的方法。尽管基于深度学习的方法取得了显著进展，但大多数现有方法仍未充分利用3D空间信息。这可能导致单个分子表示多个实际分子。为解决这些问题，我们提出了一种名为3D-Mol的新颖的基于3D结构的分子建模方法。为了准确表示完整的空间结构，我们设计了一种新颖的编码器，通过将分子分解成三个几何图形来提取3D特征。此外，我们使用20M个无标签数据通过对比学习对模型进行预训练。我们将具有相同拓扑结构的构象视为正样本对，将相反的构象视为负样本对，而权重则由构象之间的差异确定。我们在7个基准数据集上将3D-Mol与各种最先进的基准模型进行了对比。

    Molecular property prediction offers an effective and efficient approach for early screening and optimization of drug candidates. Although deep learning based methods have made notable progress, most existing works still do not fully utilize 3D spatial information. This can lead to a single molecular representation representing multiple actual molecules. To address these issues, we propose a novel 3D structure-based molecular modeling method named 3D-Mol. In order to accurately represent complete spatial structure, we design a novel encoder to extract 3D features by deconstructing the molecules into three geometric graphs. In addition, we use 20M unlabeled data to pretrain our model by contrastive learning. We consider conformations with the same topological structure as positive pairs and the opposites as negative pairs, while the weight is determined by the dissimilarity between the conformations. We compare 3D-Mol with various state-of-the-art (SOTA) baselines on 7 benchmarks and de
    
[^3]: 蛋白质序列设计的重要性加权期望最大化方法

    Importance Weighted Expectation-Maximization for Protein Sequence Design. (arXiv:2305.00386v1 [q-bio.BM])

    [http://arxiv.org/abs/2305.00386](http://arxiv.org/abs/2305.00386)

    本文提出了一种名为IsEM-Pro的方法，用于根据给定适应性标准生成蛋白质序列。在推理期间，从其潜在空间采样可以增加多样性，指导了探索高适应性区域。实验表明，相比先前最佳方法，IsEM-Pro的平均适应性得分至少高出55％，并生成了更多样化和新颖的蛋白质序列。

    

    在生物和化学领域，设计具有所需生物功能的蛋白质序列非常重要。最近的机器学习方法使用代理序列-功能模型替代昂贵的湿实验验证。本文提出了一种名为IsEM-Pro的方法，用于根据给定的适应性标准生成蛋白质序列。它是一个潜在的生成模型，并受到另外一个学习的马尔可夫随机场结构特征的增强。研究者使用蒙特卡罗期望最大化方法（MCEM）来学习这个模型。在推理期间，从其潜在空间采样可以增加多样性，而其MRF特征则指导了探索高适应性区域。在八项蛋白质序列设计任务中的实验表明，我们的IsEM-Pro的平均适应性得分至少比先前最佳方法高55％，并且生成了更多样化和新颖的蛋白质序列。

    Designing protein sequences with desired biological function is crucial in biology and chemistry. Recent machine learning methods use a surrogate sequence-function model to replace the expensive wet-lab validation. How can we efficiently generate diverse and novel protein sequences with high fitness? In this paper, we propose IsEM-Pro, an approach to generate protein sequences towards a given fitness criterion. At its core, IsEM-Pro is a latent generative model, augmented by combinatorial structure features from a separately learned Markov random fields (MRFs). We develop an Monte Carlo Expectation-Maximization method (MCEM) to learn the model. During inference, sampling from its latent space enhances diversity while its MRFs features guide the exploration in high fitness regions. Experiments on eight protein sequence design tasks show that our IsEM-Pro outperforms the previous best methods by at least 55% on average fitness score and generates more diverse and novel protein sequences.
    

