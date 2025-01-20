# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Differentially Private Secure Multiplication: Hiding Information in the Rubble of Noise.](http://arxiv.org/abs/2309.16105) | 本文研究了在分布式计算中，允许信息泄漏和近似乘法的情况下，当诚实节点数量为少数时，差分隐私和准确性之间的权衡关系。 |
| [^2] | [ThreatCrawl: A BERT-based Focused Crawler for the Cybersecurity Domain.](http://arxiv.org/abs/2304.11960) | 本文提出了一种基于BERT的焦点爬虫ThreatCrawl，使用主题建模和关键词提取技术来筛选出最可能包含有价值CTI信息的网页。 |

# 详细

[^1]: 差分隐私安全乘法：在噪声中隐藏信息

    Differentially Private Secure Multiplication: Hiding Information in the Rubble of Noise. (arXiv:2309.16105v1 [cs.IT])

    [http://arxiv.org/abs/2309.16105](http://arxiv.org/abs/2309.16105)

    本文研究了在分布式计算中，允许信息泄漏和近似乘法的情况下，当诚实节点数量为少数时，差分隐私和准确性之间的权衡关系。

    

    我们考虑私密分布式多方乘法的问题。已经确认，Shamir秘密共享编码策略可以通过Ben Or，Goldwasser，Wigderson算法（“BGW算法”）在分布式计算中实现完美的信息理论隐私。然而，完美的隐私和准确性需要一个诚实的多数，即需要$N \geq 2t+1$个计算节点以确保对抗性节点的隐私。我们通过允许一定量的信息泄漏和近似乘法来研究在诚实节点数量为少数时的编码方案，即$N< 2t+1$。我们通过使用差分隐私而不是完美隐私来测量信息泄漏，并使用均方误差度量准确性，对$N < 2t+1$的情况下的隐私-准确性权衡进行了紧密的刻画。一个新颖的技术方面是复杂地控制信息泄漏的细节。

    We consider the problem of private distributed multi-party multiplication. It is well-established that Shamir secret-sharing coding strategies can enable perfect information-theoretic privacy in distributed computation via the celebrated algorithm of Ben Or, Goldwasser and Wigderson (the "BGW algorithm"). However, perfect privacy and accuracy require an honest majority, that is, $N \geq 2t+1$ compute nodes are required to ensure privacy against any $t$ colluding adversarial nodes. By allowing for some controlled amount of information leakage and approximate multiplication instead of exact multiplication, we study coding schemes for the setting where the number of honest nodes can be a minority, that is $N< 2t+1.$ We develop a tight characterization privacy-accuracy trade-off for cases where $N < 2t+1$ by measuring information leakage using {differential} privacy instead of perfect privacy, and using the mean squared error metric for accuracy. A novel technical aspect is an intricately 
    
[^2]: ThreatCrawl：基于BERT的网络安全焦点爬虫

    ThreatCrawl: A BERT-based Focused Crawler for the Cybersecurity Domain. (arXiv:2304.11960v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2304.11960](http://arxiv.org/abs/2304.11960)

    本文提出了一种基于BERT的焦点爬虫ThreatCrawl，使用主题建模和关键词提取技术来筛选出最可能包含有价值CTI信息的网页。

    

    可公开获取的信息对于网络威胁情报（CTI）来说包含有价值的信息。这可以用于预防已经在其他系统上发生的攻击。但是，虽然有不同的标准来交流这些信息，但很多信息是以非标准化的方式在文章或博客帖子中共享的。手动浏览多个在线门户和新闻页面以发现新威胁并提取它们是一项耗时的任务。为了自动化这个扫描过程的一部分，多篇论文提出了使用自然语言处理（NLP）从文档中提取威胁指示器（IOCs）的提取器。然而，虽然这已经解决了从文档中提取信息的问题，但很少考虑搜索这些文档。本文提出了一种新的焦点爬虫ThreatCrawl，它使用双向编码器表示（BERT）搜索网络安全领域中的相关文档。ThreatCrawl使用主题建模和关键词提取技术来识别相关网站和网页，然后应用基于BERT的分类器来优先考虑最可能包含有价值CTI信息的网页。

    Publicly available information contains valuable information for Cyber Threat Intelligence (CTI). This can be used to prevent attacks that have already taken place on other systems. Ideally, only the initial attack succeeds and all subsequent ones are detected and stopped. But while there are different standards to exchange this information, a lot of it is shared in articles or blog posts in non-standardized ways. Manually scanning through multiple online portals and news pages to discover new threats and extracting them is a time-consuming task. To automize parts of this scanning process, multiple papers propose extractors that use Natural Language Processing (NLP) to extract Indicators of Compromise (IOCs) from documents. However, while this already solves the problem of extracting the information out of documents, the search for these documents is rarely considered. In this paper, a new focused crawler is proposed called ThreatCrawl, which uses Bidirectional Encoder Representations 
    

