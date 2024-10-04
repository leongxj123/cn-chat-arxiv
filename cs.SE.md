# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CodeBenchGen: Creating Scalable Execution-based Code Generation Benchmarks](https://arxiv.org/abs/2404.00566) | 提出了CodeBenchGen框架，通过利用大型语言模型将任意代码转化为评估示例，创造了一个包含大量代码示例的数据集Exec-CSN，展示了其可扩展性和实用性。 |
| [^2] | [The Model Openness Framework: Promoting Completeness and Openness for Reproducibility, Transparency and Usability in AI](https://arxiv.org/abs/2403.13784) | 提出了模型开放框架（MOF），它是一个排名分类系统，根据完整性和开放性评估机器学习模型，旨在促进完整性、开放性以及遵循开放科学原则，可以帮助准确识别模型的透明性和可重现性。 |

# 详细

[^1]: CodeBenchGen: 创建可扩展的基于执行的代码生成基准

    CodeBenchGen: Creating Scalable Execution-based Code Generation Benchmarks

    [https://arxiv.org/abs/2404.00566](https://arxiv.org/abs/2404.00566)

    提出了CodeBenchGen框架，通过利用大型语言模型将任意代码转化为评估示例，创造了一个包含大量代码示例的数据集Exec-CSN，展示了其可扩展性和实用性。

    

    为了促进在不同场景下评估代码生成系统，我们提出了CodeBenchGen，这是一个框架，可以创建可扩展的基于执行的基准，仅需要轻微的人类指导。具体来说，我们利用一个大型语言模型（LLM）将任意代码片段转化为评估示例，包括用于执行评估的测试用例。我们通过创建包含来自CodeSearchNet数据集的367个GitHub存储库中的代码修改的293个库的1,931个例子的数据集Exec-CSN，展示了我们框架的实用性。为了展示Exec-CSN中示例的复杂性和可解性，我们进行了一个人类研究，结果显示81.3%的例子可以被人类解决，61%被评为“需要努力解决”。我们对开源和专有模型进行了代码生成实验，并分析了人类和模型的性能。

    arXiv:2404.00566v1 Announce Type: cross  Abstract: To facilitate evaluation of code generation systems across diverse scenarios, we present CodeBenchGen, a framework to create scalable execution-based benchmarks that only requires light guidance from humans. Specifically, we leverage a large language model (LLM) to convert an arbitrary piece of code into an evaluation example, including test cases for execution-based evaluation. We illustrate the usefulness of our framework by creating a dataset, Exec-CSN, which includes 1,931 examples involving 293 libraries revised from code in 367 GitHub repositories taken from the CodeSearchNet dataset. To demonstrate the complexity and solvability of examples in Exec-CSN, we present a human study demonstrating that 81.3% of the examples can be solved by humans and 61% are rated as ``requires effort to solve''. We conduct code generation experiments on open-source and proprietary models and analyze the performance of both humans and models. We will
    
[^2]: 模型开放框架: 促进人工智能中的可重现性、透明度和可用性的完整性和开放性

    The Model Openness Framework: Promoting Completeness and Openness for Reproducibility, Transparency and Usability in AI

    [https://arxiv.org/abs/2403.13784](https://arxiv.org/abs/2403.13784)

    提出了模型开放框架（MOF），它是一个排名分类系统，根据完整性和开放性评估机器学习模型，旨在促进完整性、开放性以及遵循开放科学原则，可以帮助准确识别模型的透明性和可重现性。

    

    生成式人工智能（GAI）提供了前所未有的可能性，但其商业化引发了关于透明度、可重现性、偏见和安全性的担忧。许多"开源"的GAI模型缺乏完整理解和再现所必需的组件，一些采用限制性许可证，这种行为被称为"开源洗白"。我们提出了模型开放框架（MOF），这是一个根据完整性和开放性对机器学习模型进行排名分类的系统，遵循开放科学、开源、开放数据和开放获取的原则。MOF要求模型开发生命周期的特定组件被包含并根据适当的开放许可证发布。该框架旨在防止宣称自己是开放的模型被误解，指导研究人员和开发者以宽松的许可证发布所有模型组件，并帮助公司、学术界和爱好者识别可以安全采用的模型。

    arXiv:2403.13784v1 Announce Type: new  Abstract: Generative AI (GAI) offers unprecedented possibilities but its commercialization has raised concerns about transparency, reproducibility, bias, and safety. Many "open-source" GAI models lack the necessary components for full understanding and reproduction, and some use restrictive licenses, a practice known as "openwashing." We propose the Model Openness Framework (MOF), a ranked classification system that rates machine learning models based on their completeness and openness, following principles of open science, open source, open data, and open access. The MOF requires specific components of the model development lifecycle to be included and released under appropriate open licenses. This framework aims to prevent misrepresentation of models claiming to be open, guide researchers and developers in providing all model components under permissive licenses, and help companies, academia, and hobbyists identify models that can be safely adop
    

