# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Trained Without My Consent: Detecting Code Inclusion In Language Models Trained on Code](https://arxiv.org/abs/2402.09299) | 这项研究关注如何在训练代码的语言模型中检测代码包含，以解决使用这些模型进行代码审计时的版权侵权问题。 |

# 详细

[^1]: 未经本人同意的训练：在训练代码的语言模型中检测代码包含

    Trained Without My Consent: Detecting Code Inclusion In Language Models Trained on Code

    [https://arxiv.org/abs/2402.09299](https://arxiv.org/abs/2402.09299)

    这项研究关注如何在训练代码的语言模型中检测代码包含，以解决使用这些模型进行代码审计时的版权侵权问题。

    

    代码审计通过验证开发的代码是否符合标准、法规和版权保护，确保其不包含来自受保护来源的代码。在软件开发过程中，大型语言模型(LLMs)作为编码助手的出现给代码审计带来了新的挑战。训练这些模型的数据集主要来自公开可用的来源。这引发了知识产权侵权问题，因为开发者的代码已包含在数据集中。因此，使用LLMs开发的代码审计具有挑战性，因为我们无法准确确定开发过程中使用的LLM是否已经在特定的受版权保护的代码上进行了训练，因为我们无法获得这些模型的训练数据集。鉴于训练数据集的保密性，传统的代码克隆检测等方法无法确保版权侵权。

    arXiv:2402.09299v1 Announce Type: cross Abstract: Code auditing ensures that the developed code adheres to standards, regulations, and copyright protection by verifying that it does not contain code from protected sources. The recent advent of Large Language Models (LLMs) as coding assistants in the software development process poses new challenges for code auditing. The dataset for training these models is mainly collected from publicly available sources. This raises the issue of intellectual property infringement as developers' codes are already included in the dataset. Therefore, auditing code developed using LLMs is challenging, as it is difficult to reliably assert if an LLM used during development has been trained on specific copyrighted codes, given that we do not have access to the training datasets of these models. Given the non-disclosure of the training datasets, traditional approaches such as code clone detection are insufficient for asserting copyright infringement. To add
    

