# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PoisonedRAG: Knowledge Poisoning Attacks to Retrieval-Augmented Generation of Large Language Models](https://arxiv.org/abs/2402.07867) | 本论文提出了一种名为PoisonedRAG的知识污染攻击方法，用于对大型语言模型的检索增强生成进行攻击和破坏。 |

# 详细

[^1]: PoisonedRAG: 知识污染攻击对大型语言模型的检索增强生成

    PoisonedRAG: Knowledge Poisoning Attacks to Retrieval-Augmented Generation of Large Language Models

    [https://arxiv.org/abs/2402.07867](https://arxiv.org/abs/2402.07867)

    本论文提出了一种名为PoisonedRAG的知识污染攻击方法，用于对大型语言模型的检索增强生成进行攻击和破坏。

    

    大型语言模型（LLM）由于其卓越的生成能力而取得了显著的成功。尽管如此，它们也存在固有的局限性，如缺乏最新的知识和虚构。检索增强生成（RAG）是一种最先进的技术，以减轻这些限制。具体而言，对于给定的问题，RAG从知识数据库中检索相关知识，以增强LLM的输入。例如，当知识数据库中包含从维基百科收集的数百万个文本时，检索到的知识可以是与给定问题在语义上最相似的前K个文本集。因此，LLM可以利用检索到的知识作为上下文为给定问题生成答案。现有研究主要集中在改善RAG的准确性或效率，而对其安全性的探索较少。我们旨在填补这一空白。

    Large language models (LLMs) have achieved remarkable success due to their exceptional generative capabilities. Despite their success, they also have inherent limitations such as a lack of up-to-date knowledge and hallucination. Retrieval-Augmented Generation (RAG) is a state-of-the-art technique to mitigate those limitations. In particular, given a question, RAG retrieves relevant knowledge from a knowledge database to augment the input of the LLM. For instance, the retrieved knowledge could be a set of top-k texts that are most semantically similar to the given question when the knowledge database contains millions of texts collected from Wikipedia. As a result, the LLM could utilize the retrieved knowledge as the context to generate an answer for the given question. Existing studies mainly focus on improving the accuracy or efficiency of RAG, leaving its security largely unexplored. We aim to bridge the gap in this work. Particularly, we propose PoisonedRAG , a set of knowledge pois
    

