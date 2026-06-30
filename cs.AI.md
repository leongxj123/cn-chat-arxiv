# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SSM Meets Video Diffusion Models: Efficient Video Generation with Structured State Spaces](https://arxiv.org/abs/2403.07711) | 提出了一种基于状态空间模型（SSMs）的方法，用于解决使用扩散模型生成长视频序列时注意力层内存消耗增长快、限制较大的问题 |
| [^2] | [Modelling Human Values for AI Reasoning](https://arxiv.org/abs/2402.06359) | 本研究详细介绍了一个关于人类价值观的形式化模型，并展示了它在AI推理中的应用。研究通过基于社会心理学研究的关键思想，为AI系统与人类价值观的一致性提供了具体的计算表示。 |
| [^3] | [Assortment Planning with Sponsored Products](https://arxiv.org/abs/2402.06158) | 本研究主要关注零售中带有赞助产品的品类规划挑战并将其建模为组合优化任务，以实现在考虑赞助产品的情况下优化预期收入的目的。 |
| [^4] | [Granular ball computing: an efficient, robust, and interpretable adaptive multi-granularity representation and computation method.](http://arxiv.org/abs/2304.11171) | 本文提出了一种基于颗粒球计算的自适应多粒度表示和计算方法，能够提高机器学习的效率、鲁棒性和可解释性。 |

# 详细

[^1]: SSM遇上视频扩散模型: 结构化状态空间下的高效视频生成

    SSM Meets Video Diffusion Models: Efficient Video Generation with Structured State Spaces

    [https://arxiv.org/abs/2403.07711](https://arxiv.org/abs/2403.07711)

    提出了一种基于状态空间模型（SSMs）的方法，用于解决使用扩散模型生成长视频序列时注意力层内存消耗增长快、限制较大的问题

    

    鉴于图像生成通过扩散模型取得的显著成就，研究界对将这些模型扩展到视频生成表现出越来越大的兴趣。最近用于视频生成的扩散模型主要利用注意力层来提取时间特征。然而，由于注意力层的内存消耗随着序列长度的增加呈二次增长，这种限制在尝试使用扩散模型生成更长视频序列时会带来重大挑战。为了克服这一挑战，我们提出利用状态空间模型（SSMs）。由于相对于序列长度，SSMs具有线性内存消耗，最近已经引起了越来越多的关注。在实验中，我们首先通过使用UCF101这一视频生成的标准基准来评估我们基于SSM的模型。此外，为探讨SSMs在更长视频生成中的潜力，

    arXiv:2403.07711v1 Announce Type: cross  Abstract: Given the remarkable achievements in image generation through diffusion models, the research community has shown increasing interest in extending these models to video generation. Recent diffusion models for video generation have predominantly utilized attention layers to extract temporal features. However, attention layers are limited by their memory consumption, which increases quadratically with the length of the sequence. This limitation presents significant challenges when attempting to generate longer video sequences using diffusion models. To overcome this challenge, we propose leveraging state-space models (SSMs). SSMs have recently gained attention as viable alternatives due to their linear memory consumption relative to sequence length. In the experiments, we first evaluate our SSM-based model with UCF101, a standard benchmark of video generation. In addition, to investigate the potential of SSMs for longer video generation, 
    
[^2]: 为AI推理建模人类价值观

    Modelling Human Values for AI Reasoning

    [https://arxiv.org/abs/2402.06359](https://arxiv.org/abs/2402.06359)

    本研究详细介绍了一个关于人类价值观的形式化模型，并展示了它在AI推理中的应用。研究通过基于社会心理学研究的关键思想，为AI系统与人类价值观的一致性提供了具体的计算表示。

    

    当今最重要的社会挑战之一是构建其行为与人类价值观一致的AI系统，或是其使人工和人工之间相互作用的社区行为与人类价值观一致。为了解决这一挑战，我们详细介绍了一个关于人类价值观的形式化模型，以进行其明确的计算表示。据我们所知，目前尚未有人尝试过这种模型，这在考虑到将价值观与AI整合的研究数量不断增长的情况下是令人惊讶的。我们以社会心理学领域近几十年来研究人类价值观性质的大量研究为起点，致力于提供这样一个形式化模型。我们展示了这个模型如何为基于AI的价值推理提供基础装置，并证明了它在实际应用案例中的适用性。我们阐述了我们的模型如何捕捉到社会心理学研究的关键思想，并提出了未来关于人类价值观在AI中集成和跨学科研究的路线图。

    One of today's most significant societal challenges is building AI systems whose behaviour, or the behaviour it enables within communities of interacting agents (human and artificial), aligns with human values. To address this challenge, we detail a formal model of human values for their explicit computational representation. To our knowledge, this has not been attempted as yet, which is surprising given the growing volume of research integrating values within AI. Taking as our starting point the wealth of research investigating the nature of human values from social psychology over the last few decades, we set out to provide such a formal model. We show how this model can provide the foundational apparatus for AI-based reasoning over values, and demonstrate its applicability in real-world use cases. We illustrate how our model captures the key ideas from social psychology research and propose a roadmap for future integrated, and interdisciplinary, research into human values in AI. The
    
[^3]: 带有赞助产品的品类规划

    Assortment Planning with Sponsored Products

    [https://arxiv.org/abs/2402.06158](https://arxiv.org/abs/2402.06158)

    本研究主要关注零售中带有赞助产品的品类规划挑战并将其建模为组合优化任务，以实现在考虑赞助产品的情况下优化预期收入的目的。

    

    在零售行业快速发展的背景下，品类规划对于企业的成功起着至关重要的作用。随着赞助产品在在线市场的日益突出地位，零售商在有效管理产品品类方面面临新的挑战。值得注意的是，以前的品类规划研究大多忽视了赞助产品的存在及其对整体推荐效果可能产生的影响。相反，他们通常简化地假设所有产品都是有机产品或非赞助产品。这个研究空白突显了在赞助产品存在的情况下更深入探讨品类规划挑战的必要性。我们将在存在赞助产品的情况下将品类规划问题建模为组合优化任务。最终目标是计算出一种最优的品类规划方案，既能优化预期收入，又能考虑到赞助产品的存在。

    In the rapidly evolving landscape of retail, assortment planning plays a crucial role in determining the success of a business. With the rise of sponsored products and their increasing prominence in online marketplaces, retailers face new challenges in effectively managing their product assortment in the presence of sponsored products. Remarkably, previous research in assortment planning largely overlooks the existence of sponsored products and their potential impact on overall recommendation effectiveness. Instead, they commonly make the simplifying assumption that all products are either organic or non-sponsored. This research gap underscores the necessity for a more thorough investigation of the assortment planning challenge when sponsored products are in play. We formulate the assortment planning problem in the presence of sponsored products as a combinatorial optimization task. The ultimate objective is to compute an assortment plan that optimizes expected revenue while considerin
    
[^4]: 颗粒球计算：一种高效、鲁棒和可解释的自适应多粒度表示和计算方法

    Granular ball computing: an efficient, robust, and interpretable adaptive multi-granularity representation and computation method. (arXiv:2304.11171v1 [cs.LG])

    [http://arxiv.org/abs/2304.11171](http://arxiv.org/abs/2304.11171)

    本文提出了一种基于颗粒球计算的自适应多粒度表示和计算方法，能够提高机器学习的效率、鲁棒性和可解释性。

    

    人类认知具有“先大后小”的认知机制，因此具有自适应的多粒度描述能力。这导致了有效性、鲁棒性和可解释性等计算特性。本文提出了一种新的基于颗粒球计算的自适应多粒度表示和计算方法。他们将这种方法应用于几个机器学习任务，并证明其相对于其他最先进的方法的有效性。

    Human cognition has a ``large-scale first'' cognitive mechanism, therefore possesses adaptive multi-granularity description capabilities. This results in computational characteristics such as efficiency, robustness, and interpretability. Although most existing artificial intelligence learning methods have certain multi-granularity features, they do not fully align with the ``large-scale first'' cognitive mechanism. Multi-granularity granular-ball computing is an important model method developed in recent years. This method can use granular-balls of different sizes to adaptively represent and cover the sample space, and perform learning based on granular-balls. Since the number of coarse-grained "granular-ball" is smaller than the number of sample points, granular-ball computing is more efficient; the coarse-grained characteristics of granular-balls are less likely to be affected by fine-grained sample points, making them more robust; the multi-granularity structure of granular-balls ca
    

