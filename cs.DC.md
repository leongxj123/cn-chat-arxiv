# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Privacy-Aware Semantic Cache for Large Language Models](https://arxiv.org/abs/2403.02694) | MeanCache是一种面向LLMs的语义缓存，能够识别语义上相似的查询，从而减少查询成本，服务提供商负载和环境影响。 |

# 详细

[^1]: 面向大型语言模型的隐私感知语义缓存

    Privacy-Aware Semantic Cache for Large Language Models

    [https://arxiv.org/abs/2403.02694](https://arxiv.org/abs/2403.02694)

    MeanCache是一种面向LLMs的语义缓存，能够识别语义上相似的查询，从而减少查询成本，服务提供商负载和环境影响。

    

    大型语言模型（LLMs）如ChatGPT、Google Bard、Claude和Llama 2彻底改变了自然语言处理和搜索引擎动态。然而，这些模型造成了异常高的计算成本。本文介绍了MeanCache，一种用于LLMs的语义缓存，它能够识别语义上相似的查询以确定缓存命中或未命中。

    arXiv:2403.02694v1 Announce Type: cross  Abstract: Large Language Models (LLMs) like ChatGPT, Google Bard, Claude, and Llama 2 have revolutionized natural language processing and search engine dynamics. However, these models incur exceptionally high computational costs. For instance, GPT-3 consists of 175 billion parameters and inference on these models also demands billions of floating-point operations. Caching is a natural solution to reduce LLM inference costs on repeated queries. However, existing caching methods are incapable of finding semantic similarities among LLM queries, leading to unacceptable false hit-and-miss rates.   This paper introduces MeanCache, a semantic cache for LLMs that identifies semantically similar queries to determine cache hit or miss. Using MeanCache, the response to a user's semantically similar query can be retrieved from a local cache rather than re-querying the LLM, thus reducing costs, service provider load, and environmental impact. MeanCache lever
    

