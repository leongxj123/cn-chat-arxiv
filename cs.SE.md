# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Automated Test Case Generation Using Code Models and Domain Adaptation.](http://arxiv.org/abs/2308.08033) | 本研究提出了一个完全自动化的测试框架，利用开发人员编写的测试和可用的代码模型生成可编译、易读的单元测试。 |
| [^2] | [RAPGen: An Approach for Fixing Code Inefficiencies in Zero-Shot.](http://arxiv.org/abs/2306.17077) | RAPGen是一种新方法，通过在零样本情况下使用Retrieval-Augmented Prompt Generation（RAPGen）方法，即从预先构建的性能Bug修复知识库中检索提示指令并生成提示，然后在大型语言模型上生成修复方案，可以有效地解决代码低效问题。实验结果显示，在专家验证的数据集中，RAPGen在60%的情况下可以生成与开发者等效或更好的性能改进建议，其中约39%的建议完全相同。 |

# 详细

[^1]: 使用代码模型和领域适应性的自动化测试用例生成

    Automated Test Case Generation Using Code Models and Domain Adaptation. (arXiv:2308.08033v1 [cs.SE])

    [http://arxiv.org/abs/2308.08033](http://arxiv.org/abs/2308.08033)

    本研究提出了一个完全自动化的测试框架，利用开发人员编写的测试和可用的代码模型生成可编译、易读的单元测试。

    

    最先进的自动化测试生成技术，例如基于搜索的测试，通常对开发人员创建的测试用例一无所知。因此，它们通常生成的测试用例不易阅读，并且可能无法检测所有复杂缺陷，而开发人员编写的测试用例则可以。在这项研究中，我们利用基于Transformer的代码模型生成可以补充基于搜索测试生成的单元测试。具体而言，我们使用CodeT5，即最先进的大型代码模型，并对测试生成下游任务进行微调。我们使用Methods2test数据集对CodeT5进行微调，并使用Defects4j进行项目级领域适应性和评估。本研究的主要贡献是提出了一个完全自动化的测试框架，利用开发人员编写的测试和可用的代码模型生成可编译、易读的单元测试。结果显示，我们的方法可以生成新的测试用例，覆盖了已经被测试过的代码行。

    State-of-the-art automated test generation techniques, such as search-based testing, are usually ignorant about what a developer would create as a test case. Therefore, they typically create tests that are not human-readable and may not necessarily detect all types of complex bugs developer-written tests would do. In this study, we leverage Transformer-based code models to generate unit tests that can complement search-based test generation. Specifically, we use CodeT5, i.e., a state-of-the-art large code model, and fine-tune it on the test generation downstream task. For our analysis, we use the Methods2test dataset for fine-tuning CodeT5 and Defects4j for project-level domain adaptation and evaluation. The main contribution of this study is proposing a fully automated testing framework that leverages developer-written tests and available code models to generate compilable, human-readable unit tests. Results show that our approach can generate new test cases that cover lines that were
    
[^2]: RAPGen: 一种解决零样本代码低效问题的方法

    RAPGen: An Approach for Fixing Code Inefficiencies in Zero-Shot. (arXiv:2306.17077v1 [cs.SE])

    [http://arxiv.org/abs/2306.17077](http://arxiv.org/abs/2306.17077)

    RAPGen是一种新方法，通过在零样本情况下使用Retrieval-Augmented Prompt Generation（RAPGen）方法，即从预先构建的性能Bug修复知识库中检索提示指令并生成提示，然后在大型语言模型上生成修复方案，可以有效地解决代码低效问题。实验结果显示，在专家验证的数据集中，RAPGen在60%的情况下可以生成与开发者等效或更好的性能改进建议，其中约39%的建议完全相同。

    

    性能Bug是一种即使在经过充分测试的商业产品中也可能出现的非功能性问题。修复这些性能Bug是一个重要但具有挑战性的问题。在这项工作中，我们解决了这个挑战，并提出了一种名为Retrieval-Augmented Prompt Generation（RAPGen）的新方法。给定一个存在性能问题的代码片段，RAPGen首先从预先构建的之前性能Bug修复知识库中检索一个提示指令，然后使用检索到的指令生成一个提示。然后，它在零样本情况下使用这个提示在大型语言模型（如Codex）上生成一个修复方案。我们将我们的方法与各种提示变体和现有方法在性能Bug修复任务中进行了比较。我们的评估结果显示，RAPGen在60%的情况下可以生成与开发者等效或更好的性能改进建议，在经过专家验证的过去C#开发者所做的性能更改数据集中有约39%的建议完全相同。

    Performance bugs are non-functional bugs that can even manifest in well-tested commercial products. Fixing these performance bugs is an important yet challenging problem. In this work, we address this challenge and present a new approach called Retrieval-Augmented Prompt Generation (RAPGen). Given a code snippet with a performance issue, RAPGen first retrieves a prompt instruction from a pre-constructed knowledge-base of previous performance bug fixes and then generates a prompt using the retrieved instruction. It then uses this prompt on a Large Language Model (such as Codex) in zero-shot to generate a fix. We compare our approach with the various prompt variations and state of the art methods in the task of performance bug fixing. Our evaluation shows that RAPGen can generate performance improvement suggestions equivalent or better than a developer in ~60% of the cases, getting ~39% of them verbatim, in an expert-verified dataset of past performance changes made by C# developers.
    

