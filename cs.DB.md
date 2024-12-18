# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RetClean: Retrieval-Based Data Cleaning Using Foundation Models and Data Lakes.](http://arxiv.org/abs/2303.16909) | 本研究展示了使用ChatGPT对数据进行清洗的可能性，并提出了结合用户提供的数据湖的基于检索的清洗方法，同时还开发了一种在本地部署的RoBERTa模型来解决隐私问题。 |

# 详细

[^1]: RetClean: 基于基础模型与数据湖的检索式数据清洗

    RetClean: Retrieval-Based Data Cleaning Using Foundation Models and Data Lakes. (arXiv:2303.16909v1 [cs.DB])

    [http://arxiv.org/abs/2303.16909](http://arxiv.org/abs/2303.16909)

    本研究展示了使用ChatGPT对数据进行清洗的可能性，并提出了结合用户提供的数据湖的基于检索的清洗方法，同时还开发了一种在本地部署的RoBERTa模型来解决隐私问题。

    

    本研究展示了使用基础模型ChatGPT来提供数据清洗建议的可能性。但在处理企业数据或需要解释建议来源时，ChatGPT可能无法胜任。为解决这些问题，我们开发了一种基于检索的方法，该方法配合用户提供的数据湖，将数据湖的数据与ChatGPT的能力结合使用。此外，我们还开发了一种基于RoBERTa的定制化模型，用户可以在本地进行部署使用。

    Can foundation models (such as ChatGPT) clean your data? In this proposal, we demonstrate that indeed ChatGPT can assist in data cleaning by suggesting corrections for specific cells in a data table (scenario 1). However, ChatGPT may struggle with datasets it has never encountered before (e.g., local enterprise data) or when the user requires an explanation of the source of the suggested clean values. To address these issues, we developed a retrieval-based method that complements ChatGPT's power with a user-provided data lake. The data lake is first indexed, we then retrieve the top-k relevant tuples to the user's query tuple and finally leverage ChatGPT to infer the correct value (scenario 2). Nevertheless, sharing enterprise data with ChatGPT, an externally hosted model, might not be feasible for privacy reasons. To assist with this scenario, we developed a custom RoBERTa-based foundation model that can be locally deployed. By fine-tuning it on a small number of examples, it can effe
    

