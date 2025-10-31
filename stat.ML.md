# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reward Collapse in Aligning Large Language Models.](http://arxiv.org/abs/2305.17608) | 本文记录了大型语言模型训练中的奖励塌陷现象，导致在训练结束时，不同的提示生成的奖励分布相同。这主要是因为排名的目标函数无法在优化过程中考虑与提示相关的信息。 |

# 详细

[^1]: 对齐大型语言模型中的奖励塌缩现象

    Reward Collapse in Aligning Large Language Models. (arXiv:2305.17608v1 [cs.LG])

    [http://arxiv.org/abs/2305.17608](http://arxiv.org/abs/2305.17608)

    本文记录了大型语言模型训练中的奖励塌陷现象，导致在训练结束时，不同的提示生成的奖励分布相同。这主要是因为排名的目标函数无法在优化过程中考虑与提示相关的信息。

    

    大型语言模型（LLMs），如ChatGPT和GPT-4，具有非凡的能力，部分原因在于将它们与训练在人类偏好上的奖励模型对齐，这些偏好通常表示为对响应提示的排名。本文记录了奖励塌陷现象，这是一种经验观察，其中基于排名的方法导致在训练的终止阶段生成的完整奖励分布\textit{无论}\textbf{prompt是什么}都是\textit{相同的}。这种结果是不可取的，因为像“写一篇关于你最好的朋友的简短故事”这样的开放式提示应生成完成它们的连续奖励范围，而像“新西兰的首都是什么”这样的特定提示应生成高或低奖励。我们的理论调查表明，奖励塌陷主要是由于基于排名的目标函数在优化过程中未能纳入与提示相关的信息所致。

    The extraordinary capabilities of large language models (LLMs) such as ChatGPT and GPT-4 are in part unleashed by aligning them with reward models that are trained on human preferences, which are often represented as rankings of responses to prompts. In this paper, we document the phenomenon of \textit{reward collapse}, an empirical observation where the prevailing ranking-based approach results in an \textit{identical} reward distribution \textit{regardless} of the prompts during the terminal phase of training. This outcome is undesirable as open-ended prompts like ``write a short story about your best friend'' should yield a continuous range of rewards for their completions, while specific prompts like ``what is the capital of New Zealand'' should generate either high or low rewards. Our theoretical investigation reveals that reward collapse is primarily due to the insufficiency of the ranking-based objective function to incorporate prompt-related information during optimization. Thi
    

