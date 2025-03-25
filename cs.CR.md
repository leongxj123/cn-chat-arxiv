# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data Reconstruction Attacks and Defenses: A Systematic Evaluation](https://arxiv.org/abs/2402.09478) | 本研究提出了一种在联合学习环境中的强力重构攻击，可以重构中间特征，并且对大部分先前的方法表现更好。实证研究表明，在防御机制中，梯度修剪是对抗最先进攻击最有效的策略。 |
| [^2] | [Introducing Foundation Models as Surrogate Models: Advancing Towards More Practical Adversarial Attacks.](http://arxiv.org/abs/2307.06608) | 本文将对抗攻击重新设定为下游任务，通过生成图像噪声来满足新兴趋势，并将基础模型引入作为代理模型。虽然基础模型的表现不佳，但通过在特征空间中进行分析，我们发现缺乏对应的特征。 |
| [^3] | [ThreatCrawl: A BERT-based Focused Crawler for the Cybersecurity Domain.](http://arxiv.org/abs/2304.11960) | 本文提出了一种基于BERT的焦点爬虫ThreatCrawl，使用主题建模和关键词提取技术来筛选出最可能包含有价值CTI信息的网页。 |

# 详细

[^1]: 数据重构攻击与防御：一个系统评估

    Data Reconstruction Attacks and Defenses: A Systematic Evaluation

    [https://arxiv.org/abs/2402.09478](https://arxiv.org/abs/2402.09478)

    本研究提出了一种在联合学习环境中的强力重构攻击，可以重构中间特征，并且对大部分先前的方法表现更好。实证研究表明，在防御机制中，梯度修剪是对抗最先进攻击最有效的策略。

    

    重构攻击和防御对于理解机器学习中的数据泄漏问题至关重要。然而，先前的工作主要集中在梯度反转攻击的经验观察上，缺乏理论基础，并且无法区分防御方法的有用性与攻击方法的计算限制。在这项工作中，我们提出了一种在联合学习环境中的强力重构攻击。该攻击可以重构中间特征，并与大部分先前的方法相比表现更好。在这种更强的攻击下，我们从理论和实证两方面全面调查了最常见的防御方法的效果。我们的研究结果表明，在各种防御机制中，如梯度剪辑、dropout、添加噪音、局部聚合等等，梯度修剪是对抗最先进攻击最有效的策略。

    arXiv:2402.09478v1 Announce Type: cross  Abstract: Reconstruction attacks and defenses are essential in understanding the data leakage problem in machine learning. However, prior work has centered around empirical observations of gradient inversion attacks, lacks theoretical groundings, and was unable to disentangle the usefulness of defending methods versus the computational limitation of attacking methods. In this work, we propose a strong reconstruction attack in the setting of federated learning. The attack reconstructs intermediate features and nicely integrates with and outperforms most of the previous methods. On this stronger attack, we thoroughly investigate both theoretically and empirically the effect of the most common defense methods. Our findings suggest that among various defense mechanisms, such as gradient clipping, dropout, additive noise, local aggregation, etc., gradient pruning emerges as the most effective strategy to defend against state-of-the-art attacks.
    
[^2]: 将基础模型作为代理模型引入：朝着更实用的对抗攻击迈进

    Introducing Foundation Models as Surrogate Models: Advancing Towards More Practical Adversarial Attacks. (arXiv:2307.06608v1 [cs.LG])

    [http://arxiv.org/abs/2307.06608](http://arxiv.org/abs/2307.06608)

    本文将对抗攻击重新设定为下游任务，通过生成图像噪声来满足新兴趋势，并将基础模型引入作为代理模型。虽然基础模型的表现不佳，但通过在特征空间中进行分析，我们发现缺乏对应的特征。

    

    最近，无盒对抗攻击成为了最实用且具有挑战性的攻击方式，攻击者无法访问模型的架构、权重和训练数据。然而，在无盒设置中，对于代理模型选择过程的潜力和灵活性缺乏认识。受到利用基础模型解决下游任务的兴趣的启发，本文采用了1）将对抗攻击重新设定为下游任务，具体而言，是生成图像噪声以满足新兴趋势；2）将基础模型引入作为代理模型的创新思想。通过利用非鲁棒特征的概念，我们阐述了选择代理模型的两个指导原则，以解释为什么基础模型是这一角色的最佳选择。然而，矛盾地的是，我们观察到这些基础模型表现不佳。通过在特征空间中分析这种意外行为，我们归因于缺乏上述指导原则所需的特征。

    Recently, the no-box adversarial attack, in which the attacker lacks access to the model's architecture, weights, and training data, become the most practical and challenging attack setup. However, there is an unawareness of the potential and flexibility inherent in the surrogate model selection process on no-box setting. Inspired by the burgeoning interest in utilizing foundational models to address downstream tasks, this paper adopts an innovative idea that 1) recasting adversarial attack as a downstream task. Specifically, image noise generation to meet the emerging trend and 2) introducing foundational models as surrogate models. Harnessing the concept of non-robust features, we elaborate on two guiding principles for surrogate model selection to explain why the foundational model is an optimal choice for this role. However, paradoxically, we observe that these foundational models underperform. Analyzing this unexpected behavior within the feature space, we attribute the lackluster
    
[^3]: ThreatCrawl：基于BERT的网络安全焦点爬虫

    ThreatCrawl: A BERT-based Focused Crawler for the Cybersecurity Domain. (arXiv:2304.11960v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2304.11960](http://arxiv.org/abs/2304.11960)

    本文提出了一种基于BERT的焦点爬虫ThreatCrawl，使用主题建模和关键词提取技术来筛选出最可能包含有价值CTI信息的网页。

    

    可公开获取的信息对于网络威胁情报（CTI）来说包含有价值的信息。这可以用于预防已经在其他系统上发生的攻击。但是，虽然有不同的标准来交流这些信息，但很多信息是以非标准化的方式在文章或博客帖子中共享的。手动浏览多个在线门户和新闻页面以发现新威胁并提取它们是一项耗时的任务。为了自动化这个扫描过程的一部分，多篇论文提出了使用自然语言处理（NLP）从文档中提取威胁指示器（IOCs）的提取器。然而，虽然这已经解决了从文档中提取信息的问题，但很少考虑搜索这些文档。本文提出了一种新的焦点爬虫ThreatCrawl，它使用双向编码器表示（BERT）搜索网络安全领域中的相关文档。ThreatCrawl使用主题建模和关键词提取技术来识别相关网站和网页，然后应用基于BERT的分类器来优先考虑最可能包含有价值CTI信息的网页。

    Publicly available information contains valuable information for Cyber Threat Intelligence (CTI). This can be used to prevent attacks that have already taken place on other systems. Ideally, only the initial attack succeeds and all subsequent ones are detected and stopped. But while there are different standards to exchange this information, a lot of it is shared in articles or blog posts in non-standardized ways. Manually scanning through multiple online portals and news pages to discover new threats and extracting them is a time-consuming task. To automize parts of this scanning process, multiple papers propose extractors that use Natural Language Processing (NLP) to extract Indicators of Compromise (IOCs) from documents. However, while this already solves the problem of extracting the information out of documents, the search for these documents is rarely considered. In this paper, a new focused crawler is proposed called ThreatCrawl, which uses Bidirectional Encoder Representations 
    

