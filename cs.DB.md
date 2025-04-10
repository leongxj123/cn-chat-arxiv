# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimizing LLM Queries in Relational Workloads](https://arxiv.org/abs/2403.05821) | 本文研究了如何优化在关系查询中调用LLM的分析型工作负载的推理过程，发现关系查询为加速LLM推理提供了新颖的机会。 |

# 详细

[^1]: 在关系型工作负载中优化LLM查询

    Optimizing LLM Queries in Relational Workloads

    [https://arxiv.org/abs/2403.05821](https://arxiv.org/abs/2403.05821)

    本文研究了如何优化在关系查询中调用LLM的分析型工作负载的推理过程，发现关系查询为加速LLM推理提供了新颖的机会。

    

    arXiv:2403.05821v1 公告类型: 新的 摘要: 分析性数据库提供商（例如Redshift、Databricks、BigQuery）已迅速增加对通过本机用户自定义函数（UDFs）调用大型语言模型（LLMs）的支持，以帮助用户在分析型工作负载内执行自然语言任务，例如分类、实体提取和翻译。本文探讨了如何优化关系查询中调用LLM的分析工作负载的推理。我们展示了关系查询为加速LLM推理提供了新颖的机会，包括重新排序行以最大化LLM推理引擎内的键值（KV）缓存重用，重新排序行内的列以进一步。

    arXiv:2403.05821v1 Announce Type: new  Abstract: Analytical database providers (e.g., Redshift, Databricks, BigQuery) have rapidly added support for invoking Large Language Models (LLMs) through native user-defined functions (UDFs) to help users perform natural language tasks, such as classification, entity extraction, and translation, inside analytical workloads. For instance, an analyst might want to extract customer sentiments on millions of product reviews. However, LLM inference is highly expensive in both computational and economic terms: for example, an NVIDIA L4 GPU running Llama2-7B can only process 6 KB of text per second. In this paper, we explore how to optimize LLM inference for analytical workloads that invoke LLMs within relational queries. We show that relational queries present novel opportunities for accelerating LLM inference, including reordering rows to maximize key-value (KV) cache reuse within the LLM inference engine, reordering columns within a row to further i
    

