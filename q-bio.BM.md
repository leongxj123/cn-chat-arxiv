# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RiNALMo: General-Purpose RNA Language Models Can Generalize Well on Structure Prediction Tasks](https://arxiv.org/abs/2403.00043) | RiNALMo是迄今为止最大的RNA语言模型，具有650亿个参数，能够在多个下游任务上取得最先进结果，并展示了其泛化能力。 |
| [^2] | [Fast and Functional Structured Data Generators Rooted in Out-of-Equilibrium Physics.](http://arxiv.org/abs/2307.06797) | 这项研究提出了一种基于非平衡物理学的训练算法，用于解决使用能量模型生成高质量结构化数据的挑战。该方法通过改善模型的分类能力和生成速度，在多个领域取得了成功应用。 |

# 详细

[^1]: RiNALMo: 通用RNA语言模型在结构预测任务上具有良好的泛化能力

    RiNALMo: General-Purpose RNA Language Models Can Generalize Well on Structure Prediction Tasks

    [https://arxiv.org/abs/2403.00043](https://arxiv.org/abs/2403.00043)

    RiNALMo是迄今为止最大的RNA语言模型，具有650亿个参数，能够在多个下游任务上取得最先进结果，并展示了其泛化能力。

    

    arXiv:2403.00043v1 通告类型: 跨领域 摘要: 核糖核酸（RNA）在基础生物过程中扮演着各种至关重要的角色。最近，RNA已成为一个有趣的药物靶点，强调了提高我们对其结构和功能的理解的必要性。多年来，测序技术已产生了大量未标记的RNA数据，其中隐藏着重要的知识和潜力。受蛋白质语言模型成功的启发，我们引入了核糖核酸语言模型（RiNALMo）以帮助揭示RNA的隐藏密码。RiNALMo是迄今为止最大的RNA语言模型，具有650亿个参数，预先训练了来自几个可用数据库的3600万个非编码RNA序列。RiNALMo能够提取隐藏知识并隐含地捕捉RNA序列中内嵌的基本结构信息。RiNALMo在多个下游任务上实现了最先进的结果。值得注意的是，我们展示了其泛化能力

    arXiv:2403.00043v1 Announce Type: cross  Abstract: Ribonucleic acid (RNA) plays a variety of crucial roles in fundamental biological processes. Recently, RNA has become an interesting drug target, emphasizing the need to improve our understanding of its structures and functions. Over the years, sequencing technologies have produced an enormous amount of unlabeled RNA data, which hides important knowledge and potential. Motivated by the successes of protein language models, we introduce RiboNucleic Acid Language Model (RiNALMo) to help unveil the hidden code of RNA. RiNALMo is the largest RNA language model to date with $650$ million parameters pre-trained on $36$ million non-coding RNA sequences from several available databases. RiNALMo is able to extract hidden knowledge and capture the underlying structure information implicitly embedded within the RNA sequences. RiNALMo achieves state-of-the-art results on several downstream tasks. Notably, we show that its generalization capabiliti
    
[^2]: 基于非平衡物理学的快速且功能性结构化数据生成器

    Fast and Functional Structured Data Generators Rooted in Out-of-Equilibrium Physics. (arXiv:2307.06797v1 [cs.LG])

    [http://arxiv.org/abs/2307.06797](http://arxiv.org/abs/2307.06797)

    这项研究提出了一种基于非平衡物理学的训练算法，用于解决使用能量模型生成高质量结构化数据的挑战。该方法通过改善模型的分类能力和生成速度，在多个领域取得了成功应用。

    

    在这项研究中，我们解决了使用基于能量的模型在复杂结构化数据集（如人口基因组学、RNA或蛋白质序列数据）中生成高质量、标签特定数据的挑战。传统的训练方法由于马尔可夫链蒙特卡洛混合效率低下而遇到困难，这影响了合成数据的多样性并增加了生成时间。为了解决这些问题，我们使用了一种利用非平衡效应的新型训练算法。这种方法应用于受限玻尔兹曼机，提高了模型对样本的正确分类能力，并只需少数几个采样步骤即可生成高质量的合成数据。该方法的有效性通过其成功应用于四种不同类型的数据得到证明：手写数字，按大陆起源分类的人类基因组突变，酶蛋白家族的功能序列，以及特定分类法的同源RNA序列。

    In this study, we address the challenge of using energy-based models to produce high-quality, label-specific data in complex structured datasets, such as population genetics, RNA or protein sequences data. Traditional training methods encounter difficulties due to inefficient Markov chain Monte Carlo mixing, which affects the diversity of synthetic data and increases generation times. To address these issues, we use a novel training algorithm that exploits non-equilibrium effects. This approach, applied on the Restricted Boltzmann Machine, improves the model's ability to correctly classify samples and generate high-quality synthetic data in only a few sampling steps. The effectiveness of this method is demonstrated by its successful application to four different types of data: handwritten digits, mutations of human genomes classified by continental origin, functionally characterized sequences of an enzyme protein family, and homologous RNA sequences from specific taxonomies.
    

