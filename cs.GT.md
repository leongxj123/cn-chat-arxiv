# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Mechanism Design for Large Language Models.](http://arxiv.org/abs/2310.10826) | 本研究主要研究了支持AI生成内容的拍卖机制设计，通过提出令牌拍卖模型，实现了以激励兼容的方式聚合多个大型语言模型，并用于结合不同广告商的输入。这个机制设计有独特的特点，并通过制定自然的激励特性得到了验证。 |

# 详细

[^1]: 大型语言模型的机制设计

    Mechanism Design for Large Language Models. (arXiv:2310.10826v1 [cs.GT])

    [http://arxiv.org/abs/2310.10826](http://arxiv.org/abs/2310.10826)

    本研究主要研究了支持AI生成内容的拍卖机制设计，通过提出令牌拍卖模型，实现了以激励兼容的方式聚合多个大型语言模型，并用于结合不同广告商的输入。这个机制设计有独特的特点，并通过制定自然的激励特性得到了验证。

    

    我们研究拍卖机制以支持新兴的AI生成内容的格式。我们特别研究如何以激励兼容的方式聚合多个LLM。在这个问题中，每个代理对随机生成的内容的偏好被描述/编码为一个LLM。设计一个用于AI生成的广告创意的拍卖格式来结合来自不同广告商的输入是一个关键动机。我们认为，尽管这个问题通常属于机制设计的范畴，但它具有一些独特的特点。我们提出了一个通用的形式化方法——令牌拍卖模型——来研究这个问题。这个模型的一个关键特点是它以令牌为单位进行操作，并允许LLM代理通过一维出价影响生成的内容。我们首先探讨了一个强大的拍卖设计方法，其中我们假设的是代理人的偏好涉及到结果分布的偏序。我们制定了两个自然的激励特性，并证明了这些特性的重要性。

    We investigate auction mechanisms to support the emerging format of AI-generated content. We in particular study how to aggregate several LLMs in an incentive compatible manner. In this problem, the preferences of each agent over stochastically generated contents are described/encoded as an LLM. A key motivation is to design an auction format for AI-generated ad creatives to combine inputs from different advertisers. We argue that this problem, while generally falling under the umbrella of mechanism design, has several unique features. We propose a general formalism -- the token auction model -- for studying this problem. A key feature of this model is that it acts on a token-by-token basis and lets LLM agents influence generated contents through single dimensional bids.  We first explore a robust auction design approach, in which all we assume is that agent preferences entail partial orders over outcome distributions. We formulate two natural incentive properties, and show that these 
    

