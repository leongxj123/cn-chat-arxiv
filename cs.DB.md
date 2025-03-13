# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LLM-SQL-Solver: Can LLMs Determine SQL Equivalence?.](http://arxiv.org/abs/2312.10321) | 本研究探讨了LLM是否能够确定两个SQL查询的等价关系，并提出了两种提示技术来帮助LLM生成高质量的响应。 |

# 详细

[^1]: LLM-SQL-Solver: LLM能够确定SQL等价关系吗？

    LLM-SQL-Solver: Can LLMs Determine SQL Equivalence?. (arXiv:2312.10321v2 [cs.DB] UPDATED)

    [http://arxiv.org/abs/2312.10321](http://arxiv.org/abs/2312.10321)

    本研究探讨了LLM是否能够确定两个SQL查询的等价关系，并提出了两种提示技术来帮助LLM生成高质量的响应。

    

    判断两个SQL查询之间的等价关系是数据管理和SQL生成中的一个基本问题，具有许多实际应用（即，在文本到SQL任务中评估生成的SQL查询的质量）。虽然研究界多年来一直在考虑SQL的等价性，但它存在相当大的困难，并且没有完整的解决方案。最近，大型语言模型（LLMs）在对话、问答和解决数学问题方面展现出强大的推理能力。在本文中，我们研究了LLMs是否可以用于确定两个SQL查询的等价性（语义等价和宽松等价）。为了帮助LLMs生成高质量的响应，我们提出了两种提示技术：Miniature & Mull和Explain & Compare。前一种技术被用于评估语义等价性，它要求LLMs在简单的数据库实例上执行查询，然后探索是否存在反例。

    Judging the equivalence between two SQL queries is a fundamental problem with many practical applications in data management and SQL generation (i.e., evaluating the quality of generated SQL queries in text-to-SQL task). While the research community has reasoned about SQL equivalence for decades, it poses considerable difficulties and no complete solutions exist. Recently, Large Language Models (LLMs) have shown strong reasoning capability in conversation, question answering and solving mathematics challenges. In this paper, we study if LLMs can be used to determine the equivalence between SQL queries under two notions of SQL equivalence (semantic equivalence and relaxed equivalence). To assist LLMs in generating high quality responses, we present two prompting techniques: Miniature & Mull and Explain & Compare. The former technique is used to evaluate the semantic equivalence in which it asks LLMs to execute a query on a simple database instance and then explore if a counterexample ex
    

