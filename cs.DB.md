# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Structure Guided Large Language Model for SQL Generation](https://arxiv.org/abs/2402.13284) | 通过引入结构信息，提出了一个结构引导的SQL生成模型，以改善大型语言模型生成SQL的准确性和可执行性。 |

# 详细

[^1]: 结构引导的大型语言模型用于SQL生成

    Structure Guided Large Language Model for SQL Generation

    [https://arxiv.org/abs/2402.13284](https://arxiv.org/abs/2402.13284)

    通过引入结构信息，提出了一个结构引导的SQL生成模型，以改善大型语言模型生成SQL的准确性和可执行性。

    

    生成准确的结构化查询语言（SQL）是一个长期存在的问题，特别是在将用户的语义查询与结构化数据库匹配，然后生成结构化SQL方面。现有模型通常将查询和数据库模式输入到LLM中，并依赖LLM执行语义-结构匹配并生成结构化SQL。然而，这种解决方案忽略了用户查询和数据库中的结构信息，而这些信息可以用来增强结构化SQL的生成。这一疏忽可能导致不准确或无法执行的SQL生成。为了充分利用结构，我们提出了一个结构到SQL的框架，利用固有的结构信息来改善LLM的SQL生成。具体地，我们介绍了我们的结构引导SQL（SGU-SQL）生成模型。

    arXiv:2402.13284v1 Announce Type: cross  Abstract: Generating accurate Structured Querying Language (SQL) is a long-standing problem, especially in matching users' semantic queries with structured databases and then generating structured SQL. Existing models typically input queries and database schemas into the LLM and rely on the LLM to perform semantic-structure matching and generate structured SQL. However, such solutions overlook the structural information within user queries and databases, which can be utilized to enhance the generation of structured SQL. This oversight can lead to inaccurate or unexecutable SQL generation. To fully exploit the structure, we propose a structure-to-SQL framework, which leverages the inherent structure information to improve the SQL generation of LLMs. Specifically, we introduce our Structure Guided SQL~(SGU-SQL) generation model. SGU-SQL first links user queries and databases in a structure-enhanced manner. It then decomposes complicated linked str
    

