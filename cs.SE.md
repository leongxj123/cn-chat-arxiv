# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Vulnerability Detection with Code Language Models: How Far Are We?](https://arxiv.org/abs/2403.18624) | 我们提出了一个新的数据集PrimeVul，用于训练和评估代码语言模型进行漏洞检测，通过引入一套新颖的数据标记技术，实现了与人工验证基准相当的标签准确性，显著扩大了数据集。 |
| [^2] | [CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X.](http://arxiv.org/abs/2303.17568) | CodeGeeX是一个多语言模型，具有130亿参数，用于代码生成。经过广泛的实验证明，CodeGeeX在HumanEval-X上的代码生成和翻译任务中表现优异。此外，CodeGeeX可以将程序员的生产力提高22%。 |

# 详细

[^1]: 使用代码语言模型进行漏洞检测：我们离目标有多远？

    Vulnerability Detection with Code Language Models: How Far Are We?

    [https://arxiv.org/abs/2403.18624](https://arxiv.org/abs/2403.18624)

    我们提出了一个新的数据集PrimeVul，用于训练和评估代码语言模型进行漏洞检测，通过引入一套新颖的数据标记技术，实现了与人工验证基准相当的标签准确性，显著扩大了数据集。

    

    在代码语言模型（code LMs）和漏洞检测备受关注的背景下，我们研究了代码语言模型用于检测漏洞的有效性。我们的分析揭示了现有漏洞数据集存在的重大缺陷，包括数据质量低、标签准确性差以及高重复率，导致在现实漏洞检测场景中模型性能不可靠。此外，这些数据集使用的评估方法也不能代表真实世界的漏洞检测情景。

    arXiv:2403.18624v1 Announce Type: cross  Abstract: In the context of the rising interest in code language models (code LMs) and vulnerability detection, we study the effectiveness of code LMs for detecting vulnerabilities. Our analysis reveals significant shortcomings in existing vulnerability datasets, including poor data quality, low label accuracy, and high duplication rates, leading to unreliable model performance in realistic vulnerability detection scenarios. Additionally, the evaluation methods used with these datasets are not representative of real-world vulnerability detection.   To address these challenges, we introduce PrimeVul, a new dataset for training and evaluating code LMs for vulnerability detection. PrimeVul incorporates a novel set of data labeling techniques that achieve comparable label accuracy to human-verified benchmarks while significantly expanding the dataset. It also implements a rigorous data de-duplication and chronological data splitting strategy to miti
    
[^2]: CodeGeeX：多语言评估下的预训练代码生成模型

    CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X. (arXiv:2303.17568v1 [cs.LG])

    [http://arxiv.org/abs/2303.17568](http://arxiv.org/abs/2303.17568)

    CodeGeeX是一个多语言模型，具有130亿参数，用于代码生成。经过广泛的实验证明，CodeGeeX在HumanEval-X上的代码生成和翻译任务中表现优异。此外，CodeGeeX可以将程序员的生产力提高22%。

    

    大型预训练代码生成模型（如OpenAI Codex）可以生成正确语法和功能的代码，使程序员的编码更加高效，使我们对人工智能的追求更加贴近现实。本文介绍了CodeGeeX，一个具有130亿参数的多语言模型，用于代码生成。CodeGeeX在2022年6月时基于23种编程语言的8500亿令牌进行了预训练。我们的广泛实验表明，CodeGeeX在HumanEval-X上的代码生成和翻译任务中均优于规模相似的多语言代码模型。在HumanEval（仅限Python）的基础上，我们开发了HumanEval-X基准测试，通过手写C ++、Java、JavaScript和Go的解决方案来评估多语言模型。此外，我们在Visual Studio Code、JetBrains和Cloud Studio上构建了基于CodeGeeX的扩展，每周为数以万计的活跃用户生成47亿令牌。我们的用户研究表明，CodeGeeX可以将程序员的生产力提高22%。

    Large pre-trained code generation models, such as OpenAI Codex, can generate syntax- and function-correct code, making the coding of programmers more productive and our pursuit of artificial general intelligence closer. In this paper, we introduce CodeGeeX, a multilingual model with 13 billion parameters for code generation. CodeGeeX is pre-trained on 850 billion tokens of 23 programming languages as of June 2022. Our extensive experiments suggest that CodeGeeX outperforms multilingual code models of similar scale for both the tasks of code generation and translation on HumanEval-X. Building upon HumanEval (Python only), we develop the HumanEval-X benchmark for evaluating multilingual models by hand-writing the solutions in C++, Java, JavaScript, and Go. In addition, we build CodeGeeX-based extensions on Visual Studio Code, JetBrains, and Cloud Studio, generating 4.7 billion tokens for tens of thousands of active users per week. Our user study demonstrates that CodeGeeX can help to inc
    

