# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Exploring the Impact of the Output Format on the Evaluation of Large Language Models for Code Translation](https://arxiv.org/abs/2403.17214) | 本研究实证分析了11种流行的专门调整的大型语言模型在五种语言上生成的输出，发现其中26.4%到73.7%的代码翻译需要后处理。 |

# 详细

[^1]: 探究输出格式对大型语言模型在代码翻译评估中的影响

    Exploring the Impact of the Output Format on the Evaluation of Large Language Models for Code Translation

    [https://arxiv.org/abs/2403.17214](https://arxiv.org/abs/2403.17214)

    本研究实证分析了11种流行的专门调整的大型语言模型在五种语言上生成的输出，发现其中26.4%到73.7%的代码翻译需要后处理。

    

    编程语言之间的代码翻译是软件工程中长期存在且至关重要的任务，有助于现代化遗留系统，确保跨平台兼容性，提升软件性能。随着大型语言模型（LLMs）及其在代码翻译中的应用的最新进展，对这些模型进行全面评估的需求越来越强烈。在本研究中，我们在五种语言（包括C、C++、Go、Java和Python）上，从1B到46.7B的参数范围内对十一种流行的专门调整的LLMs生成的输出进行了实证分析，并涵盖3820个翻译对。我们的分析发现，在我们评估的LLMs中，26.4%到73.7%的代码翻译需要后处理，因为这些翻译通常包含代码、引号和文本的混合，而不仅仅是纯源代码。忽视这些模型的输出格式可能不经意间导致

    arXiv:2403.17214v1 Announce Type: cross  Abstract: Code translation between programming languages is a long-existing and critical task in software engineering, facilitating the modernization of legacy systems, ensuring cross-platform compatibility, and enhancing software performance. With the recent advances in large language models (LLMs) and their applications to code translation, there is an increasing need for comprehensive evaluation of these models. In this study, we empirically analyze the generated outputs of eleven popular instruct-tuned LLMs with parameters ranging from 1B up to 46.7B on 3,820 translation pairs across five languages, including C, C++, Go, Java, and Python. Our analysis found that between 26.4% and 73.7% of code translations produced by our evaluated LLMs necessitate post-processing, as these translations often include a mix of code, quotes, and text rather than being purely source code. Overlooking the output format of these models can inadvertently lead to u
    

