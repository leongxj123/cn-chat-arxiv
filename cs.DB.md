# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [OmniPred: Language Models as Universal Regressors](https://arxiv.org/abs/2402.14547) | 本文提出了OmniPred框架，用于训练语言模型作为通用的端到端回归器，实验证明，在多个任务上训练时，语言模型能够显著优于传统回归模型。 |

# 详细

[^1]: OmniPred：语言模型作为通用回归器

    OmniPred: Language Models as Universal Regressors

    [https://arxiv.org/abs/2402.14547](https://arxiv.org/abs/2402.14547)

    本文提出了OmniPred框架，用于训练语言模型作为通用的端到端回归器，实验证明，在多个任务上训练时，语言模型能够显著优于传统回归模型。

    

    在实验设计的广阔领域中，回归一直是一个强大的工具，可以准确预测系统或模型在给定一组参数的情况下的结果指标，但传统上只限于适用于特定任务的方法。在本文中，我们提出了OmniPred，这是一个用于训练语言模型作为通用端到端回归器的框架，使用来自多样真实世界实验的$(x,y)$评估数据。通过使用源自Google Vizier的数据，这是世界上最大的黑盒优化数据库之一，我们的大量实验表明，仅通过数学参数和值的文本表示，语言模型能够进行非常精确的数值回归，如果有机会训练多个任务，则可以显著优于传统的回归模型。

    arXiv:2402.14547v1 Announce Type: cross  Abstract: Over the broad landscape of experimental design, regression has been a powerful tool to accurately predict the outcome metrics of a system or model given a set of parameters, but has been traditionally restricted to methods which are only applicable to a specific task. In this paper, we propose OmniPred, a framework for training language models as universal end-to-end regressors over $(x,y)$ evaluation data from diverse real world experiments. Using data sourced from Google Vizier, one of the largest blackbox optimization databases in the world, our extensive experiments demonstrate that through only textual representations of mathematical parameters and values, language models are capable of very precise numerical regression, and if given the opportunity to train over multiple tasks, can significantly outperform traditional regression models.
    

