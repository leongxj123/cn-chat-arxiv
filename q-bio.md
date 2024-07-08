# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [$\textit{L+M-24}$: Building a Dataset for Language + Molecules @ ACL 2024](https://arxiv.org/abs/2403.00791) | 这个论文介绍了$\textit{L+M-24}$数据集，该数据集专为ACL 2024年的语言+分子研讨会共享任务而设计，重点关注自然语言在分子设计中的三个关键优势：组合性、功能性和抽象性。 |
| [^2] | [When Representations Align: Universality in Representation Learning Dynamics](https://arxiv.org/abs/2402.09142) | 本研究提出了一个有效的表示学习理论，该理论假设了编码映射和解码映射为任意光滑函数，并且能够描述复杂且大型架构中的表示学习动力学。 |
| [^3] | [Gene Set Summarization using Large Language Models.](http://arxiv.org/abs/2305.13338) | 该论文介绍了一种使用大型语言模型来对基因集进行函数概括的方法，名为SPINDOCTOR，可以提供比传统方法更好的性能和可解释性。 |
| [^4] | [FakET: Simulating Cryo-Electron Tomograms with Neural Style Transfer.](http://arxiv.org/abs/2304.02011) | 本文提出了一种使用加性噪声和神经风格迁移技术来模拟电子显微镜正向算子，以解决深度学习方法需要大量训练数据集的问题。该方法在粒子定位和分类任务上表现良好。 |

# 详细

[^1]: $\textit{L+M-24}$：在ACL 2024年为语言+分子构建数据集

    $\textit{L+M-24}$: Building a Dataset for Language + Molecules @ ACL 2024

    [https://arxiv.org/abs/2403.00791](https://arxiv.org/abs/2403.00791)

    这个论文介绍了$\textit{L+M-24}$数据集，该数据集专为ACL 2024年的语言+分子研讨会共享任务而设计，重点关注自然语言在分子设计中的三个关键优势：组合性、功能性和抽象性。

    

    语言-分子模型已成为分子发现和理解的一个激动人心的方向。然而，由于分子-语言对数据集的稀缺性，训练这些模型具有挑战性。目前已发布的数据集有以下几种类型：1) 小规模且从现有数据库中抓取，2) 大规模但嘈杂且通过在科学文献上执行实体链接来构建，3) 通过将属性预测数据集转换为自然语言使用模板而构建。在本文档中，我们详细介绍了为ACL 2024年的语言+分子研讨会共享任务创建的$\textit{L+M-24}$数据集。特别地，$\textit{L+M-24}$旨在集中关注自然语言在分子设计中的三项关键优势：组合性、功能性和抽象性。

    arXiv:2403.00791v1 Announce Type: cross  Abstract: Language-molecule models have emerged as an exciting direction for molecular discovery and understanding. However, training these models is challenging due to the scarcity of molecule-language pair datasets. At this point, datasets have been released which are 1) small and scraped from existing databases, 2) large but noisy and constructed by performing entity linking on the scientific literature, and 3) built by converting property prediction datasets to natural language using templates. In this document, we detail the $\textit{L+M-24}$ dataset, which has been created for the Language + Molecules Workshop shared task at ACL 2024. In particular, $\textit{L+M-24}$ is designed to focus on three key benefits of natural language in molecule design: compositionality, functionality, and abstraction.
    
[^2]: 当表示对齐时：表示学习动力学中的普遍性

    When Representations Align: Universality in Representation Learning Dynamics

    [https://arxiv.org/abs/2402.09142](https://arxiv.org/abs/2402.09142)

    本研究提出了一个有效的表示学习理论，该理论假设了编码映射和解码映射为任意光滑函数，并且能够描述复杂且大型架构中的表示学习动力学。

    

    深度神经网络有许多不同的大小和结构。架构的选择，结合数据集和学习算法，普遍认为影响了学习到的神经表示。然而，最近的研究结果显示，不同的架构学习到的表示具有惊人的定性相似性。在这里，我们在将输入到隐藏表示的编码映射和从表示到输出的解码映射都是任意光滑函数的假设下，推导了表示学习的有效理论。在复杂和大型架构的情况下，隐藏表示没有被参数化强约束，该理论概括了表示学习动力学。我们通过实验证明，这个有效理论描述了具有不同激活函数和架构的深度网络中表示学习动力学的一些方面。

    arXiv:2402.09142v1 Announce Type: new Abstract: Deep neural networks come in many sizes and architectures. The choice of architecture, in conjunction with the dataset and learning algorithm, is commonly understood to affect the learned neural representations. Yet, recent results have shown that different architectures learn representations with striking qualitative similarities. Here we derive an effective theory of representation learning under the assumption that the encoding map from input to hidden representation and the decoding map from representation to output are arbitrary smooth functions. This theory schematizes representation learning dynamics in the regime of complex, large architectures, where hidden representations are not strongly constrained by the parametrization. We show through experiments that the effective theory describes aspects of representation learning dynamics across a range of deep networks with different activation functions and architectures, and exhibits 
    
[^3]: 使用大型语言模型进行基因集概括

    Gene Set Summarization using Large Language Models. (arXiv:2305.13338v1 [q-bio.GN])

    [http://arxiv.org/abs/2305.13338](http://arxiv.org/abs/2305.13338)

    该论文介绍了一种使用大型语言模型来对基因集进行函数概括的方法，名为SPINDOCTOR，可以提供比传统方法更好的性能和可解释性。

    

    分子生物学家经常解释从高通量实验和计算分析中获得的基因列表。这通常是通过统计富集分析来完成的，该分析测量与基因或其属性相关的生物功能术语的过度或欠表示程度，基于知识库（KB）（例如Gene Ontology（GO））中的编译断言。解释基因列表也可以被构建为一个文本概括任务，利用大型语言模型（LLMs）进行，可能直接利用科学文本并避免依赖KB。我们开发了SPINDOCTOR（稳定的提示插值的受控术语的自然语言描述的结构化报告模板），一种使用GPT模型执行基因集函数概括的方法，作为标准富集分析的补充。该方法可以使用不同的基因功能信息来源：（1）从鉴定的本体KB注释中获得的结构化文本，（2）从文本挖掘中推断的本体术语，以及（3）直接从非结构化文本中获得的术语。我们在一个1813个基因集的基准数据集上评估了SPINDOCTOR，并展示了使用GPT模型显著改善了现有方法的性能，同时也提高了可解释性，因为它能够生成人类可读的基因功能摘要。

    Molecular biologists frequently interpret gene lists derived from high-throughput experiments and computational analysis. This is typically done as a statistical enrichment analysis that measures the over- or under-representation of biological function terms associated with genes or their properties, based on curated assertions from a knowledge base (KB) such as the Gene Ontology (GO). Interpreting gene lists can also be framed as a textual summarization task, enabling the use of Large Language Models (LLMs), potentially utilizing scientific texts directly and avoiding reliance on a KB.  We developed SPINDOCTOR (Structured Prompt Interpolation of Natural Language Descriptions of Controlled Terms for Ontology Reporting), a method that uses GPT models to perform gene set function summarization as a complement to standard enrichment analysis. This method can use different sources of gene functional information: (1) structured text derived from curated ontological KB annotations, (2) ontol
    
[^4]: FakET: 利用神经风格迁移模拟冷冻电子断层图像

    FakET: Simulating Cryo-Electron Tomograms with Neural Style Transfer. (arXiv:2304.02011v1 [cs.LG])

    [http://arxiv.org/abs/2304.02011](http://arxiv.org/abs/2304.02011)

    本文提出了一种使用加性噪声和神经风格迁移技术来模拟电子显微镜正向算子，以解决深度学习方法需要大量训练数据集的问题。该方法在粒子定位和分类任务上表现良好。

    

    粒子定位和分类是计算显微学中最基本的问题之一。近年来，深度学习方法在这些任务中取得了巨大成功。这些监督式学习方法的一个关键缺点是它们需要大量的训练数据集，通常是与模拟透射电子显微镜物理的复杂数值正向模型中的粒子模型结合生成的。这些模型的计算机实现非常耗费计算资源，限制了它们的适用范围。本文提出了一种基于加性噪声和神经风格迁移技术模拟电子显微镜正向算子的简单方法。我们使用目前最先进的已经建立的状态之一对定位和分类任务进行评估，显示出与基准测试相当的性能。与以前的方法不同，我们的方法加速了运算，显著减少了计算成本。

    Particle localization and -classification constitute two of the most fundamental problems in computational microscopy. In recent years, deep learning based approaches have been introduced for these tasks with great success. A key shortcoming of these supervised learning methods is their need for large training data sets, typically generated from particle models in conjunction with complex numerical forward models simulating the physics of transmission electron microscopes. Computer implementations of such forward models are computationally extremely demanding and limit the scope of their applicability. In this paper we propose a simple method for simulating the forward operator of an electron microscope based on additive noise and Neural Style Transfer techniques. We evaluate the method on localization and classification tasks using one of the established state-of-the-art architectures showing performance on par with the benchmark. In contrast to previous approaches, our method acceler
    

