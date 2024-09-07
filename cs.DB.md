# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Aligning Large Language Models to a Domain-specific Graph Database](https://arxiv.org/abs/2402.16567) | 该论文提出了一种将大型语言模型对齐到特定领域的图数据库的方法，通过利用ChatGPT生成NL-GQL数据对并微调LLMs，实现了两者之间的对齐。 |

# 详细

[^1]: 将大型语言模型对齐到特定领域的图数据库

    Aligning Large Language Models to a Domain-specific Graph Database

    [https://arxiv.org/abs/2402.16567](https://arxiv.org/abs/2402.16567)

    该论文提出了一种将大型语言模型对齐到特定领域的图数据库的方法，通过利用ChatGPT生成NL-GQL数据对并微调LLMs，实现了两者之间的对齐。

    

    图数据库（Graph DB）被广泛应用于金融、社交网络和医药等各个领域。然而，将自然语言（NL）转换为图查询语言（GQL），通常称为NL2GQL，由于其固有复杂性和专业化特性而变得具有挑战性。一些方法试图利用大型语言模型（LLMs）来解决类似的任务，如文本转SQL。然而，在特定领域的NL2GQL任务中，缺乏特定领域的NL-GQL数据对使得难以建立LLMs和图数据库之间的对齐。为了解决这一挑战，我们提出了一个明确定义的流水线。具体地，我们利用ChatGPT基于给定的图数据库自我生成NL-GQL数据对。然后，我们使用创建的数据来对LLMs进行微调，从而实现LLMs与图数据库之间的对齐。此外，在推断过程中，我们提出了一种提取相关信息的方法。

    arXiv:2402.16567v1 Announce Type: new  Abstract: Graph Databases (Graph DB) are widely applied in various fields, including finance, social networks, and medicine. However, translating Natural Language (NL) into the Graph Query Language (GQL), commonly known as NL2GQL, proves to be challenging due to its inherent complexity and specialized nature. Some approaches have sought to utilize Large Language Models (LLMs) to address analogous tasks like text2SQL. Nevertheless, when it comes to NL2GQL taskson a particular domain, the absence of domain-specific NL-GQL data pairs makes it difficult to establish alignment between LLMs and the graph DB. To address this challenge, we propose a well-defined pipeline. Specifically, we utilize ChatGPT to create NL-GQL data pairs based on the given graph DB with self-instruct. Then, we use the created data to fine-tune LLMs, thereby achieving alignment between LLMs and the graph DB. Additionally, during inference, we propose a method that extracts relev
    

