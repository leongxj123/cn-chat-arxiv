# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Certifying LLM Safety against Adversarial Prompting.](http://arxiv.org/abs/2309.02705) | 本研究提出了首个具有可验证安全保证的框架——消除和检查，用于对抗敌对提示。通过逐个消除标记并使用安全过滤器检查生成的子序列，确保任何敌对修改的有害输入提示都能被正确标识为有害。 |

# 详细

[^1]: 证明LLM对抗敌对提示的安全性

    Certifying LLM Safety against Adversarial Prompting. (arXiv:2309.02705v1 [cs.CL])

    [http://arxiv.org/abs/2309.02705](http://arxiv.org/abs/2309.02705)

    本研究提出了首个具有可验证安全保证的框架——消除和检查，用于对抗敌对提示。通过逐个消除标记并使用安全过滤器检查生成的子序列，确保任何敌对修改的有害输入提示都能被正确标识为有害。

    

    为了确保语言模型的输出安全，公开使用的大型语言模型（LLM）引入了所谓的“模型对齐”防护措施。一个对齐的语言模型应该拒绝用户的请求生成有害内容。然而，这种安全措施容易受到敌对提示的攻击，敌对提示包含恶意设计的标记序列，以规避模型的安全防护并导致生成有害内容。在这项工作中，我们介绍了可验证安全保证的第一个对抗敌对提示的框架——消除和检查。我们逐个消除标记，并使用安全过滤器检查生成的子序列。如果安全过滤器检测到任何子序列或输入提示有害，我们的过程将将输入提示标记为有害。这保证了对于某个特定大小的有害输入提示的任何敌对修改也将被标记为有害。我们对抗三种攻击模式：i)敌对后缀，即附加敌对序列…

    Large language models (LLMs) released for public use incorporate guardrails to ensure their output is safe, often referred to as "model alignment." An aligned language model should decline a user's request to produce harmful content. However, such safety measures are vulnerable to adversarial prompts, which contain maliciously designed token sequences to circumvent the model's safety guards and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework to defend against adversarial prompts with verifiable safety guarantees. We erase tokens individually and inspect the resulting subsequences using a safety filter. Our procedure labels the input prompt as harmful if any subsequences or the input prompt are detected as harmful by the filter. This guarantees that any adversarial modification of a harmful prompt up to a certain size is also labeled harmful. We defend against three attack modes: i) adversarial suffix, which appends an adversarial seq
    

