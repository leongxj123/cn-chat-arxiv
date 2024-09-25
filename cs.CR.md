# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack](https://arxiv.org/abs/2404.01833) | 提出了一种名为Crescendo的新型多回合越狱攻击方法，通过看似良性的对话方式逐渐升级与模型的交互，成功突破了大型语言模型的限制。 |

# 详细

[^1]: 伟大，现在写一篇关于此的文章：Crescendo多回合LLM越狱攻击

    Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack

    [https://arxiv.org/abs/2404.01833](https://arxiv.org/abs/2404.01833)

    提出了一种名为Crescendo的新型多回合越狱攻击方法，通过看似良性的对话方式逐渐升级与模型的交互，成功突破了大型语言模型的限制。

    

    大型语言模型（LLMs）的流行程度大幅上升，并且越来越多地被应用于多个领域。这些LLMs在设计上避免涉及非法或不道德的话题，以避免对负责任的AI造成伤害。然而，最近出现了一系列攻击，被称为“越狱”，旨在突破这种对齐。直观地说，越狱攻击旨在缩小模型能做的与愿意做的之间的差距。本文介绍了一种名为Crescendo的新型越狱攻击。与现有的越狱方法不同，Crescendo是一种多回合越狱，以一种看似良性的方式与模型进行交互。它从有关手头任务的一般提示或问题开始，然后逐渐升级对话，引用模型的回复，逐渐导致成功越狱。我们在包括ChatGPT、Gemini Pr在内的各种公共系统上评估了Crescendo。

    arXiv:2404.01833v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have risen significantly in popularity and are increasingly being adopted across multiple applications. These LLMs are heavily aligned to resist engaging in illegal or unethical topics as a means to avoid contributing to responsible AI harms. However, a recent line of attacks, known as "jailbreaks", seek to overcome this alignment. Intuitively, jailbreak attacks aim to narrow the gap between what the model can do and what it is willing to do. In this paper, we introduce a novel jailbreak attack called Crescendo. Unlike existing jailbreak methods, Crescendo is a multi-turn jailbreak that interacts with the model in a seemingly benign manner. It begins with a general prompt or question about the task at hand and then gradually escalates the dialogue by referencing the model's replies, progressively leading to a successful jailbreak. We evaluate Crescendo on various public systems, including ChatGPT, Gemini Pr
    

