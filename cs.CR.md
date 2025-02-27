# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack](https://arxiv.org/abs/2404.01833) | 提出了一种名为Crescendo的新型多回合越狱攻击方法，通过看似良性的对话方式逐渐升级与模型的交互，成功突破了大型语言模型的限制。 |
| [^2] | [Syntactic Ghost: An Imperceptible General-purpose Backdoor Attacks on Pre-trained Language Models](https://arxiv.org/abs/2402.18945) | 论文提出了一种名为Syntactic Ghost的新方法，实现了对预训练语言模型进行无感知和通用的后门植入。 |
| [^3] | [Permute-and-Flip: An optimally robust and watermarkable decoder for LLMs](https://arxiv.org/abs/2402.05864) | 提出了一种名为Permute-and-Flip（PF）解码器，其具有最佳的鲁棒性和质量-鲁棒性的 tradeoff，且比采样方法更好。还设计了一种针对PF解码器的水印方案，能够保持样本的分布不变，并实现任意低的假阳性率和高的召回率。实验证明PF解码器在困惑度方面明显优于朴素采样，为LLM解码提供了一种有希望的新方法。 |

# 详细

[^1]: 伟大，现在写一篇关于此的文章：Crescendo多回合LLM越狱攻击

    Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack

    [https://arxiv.org/abs/2404.01833](https://arxiv.org/abs/2404.01833)

    提出了一种名为Crescendo的新型多回合越狱攻击方法，通过看似良性的对话方式逐渐升级与模型的交互，成功突破了大型语言模型的限制。

    

    大型语言模型（LLMs）的流行程度大幅上升，并且越来越多地被应用于多个领域。这些LLMs在设计上避免涉及非法或不道德的话题，以避免对负责任的AI造成伤害。然而，最近出现了一系列攻击，被称为“越狱”，旨在突破这种对齐。直观地说，越狱攻击旨在缩小模型能做的与愿意做的之间的差距。本文介绍了一种名为Crescendo的新型越狱攻击。与现有的越狱方法不同，Crescendo是一种多回合越狱，以一种看似良性的方式与模型进行交互。它从有关手头任务的一般提示或问题开始，然后逐渐升级对话，引用模型的回复，逐渐导致成功越狱。我们在包括ChatGPT、Gemini Pr在内的各种公共系统上评估了Crescendo。

    arXiv:2404.01833v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have risen significantly in popularity and are increasingly being adopted across multiple applications. These LLMs are heavily aligned to resist engaging in illegal or unethical topics as a means to avoid contributing to responsible AI harms. However, a recent line of attacks, known as "jailbreaks", seek to overcome this alignment. Intuitively, jailbreak attacks aim to narrow the gap between what the model can do and what it is willing to do. In this paper, we introduce a novel jailbreak attack called Crescendo. Unlike existing jailbreak methods, Crescendo is a multi-turn jailbreak that interacts with the model in a seemingly benign manner. It begins with a general prompt or question about the task at hand and then gradually escalates the dialogue by referencing the model's replies, progressively leading to a successful jailbreak. We evaluate Crescendo on various public systems, including ChatGPT, Gemini Pr
    
[^2]: Syntactic Ghost：一种对预训练语言模型进行的无感知通用后门攻击

    Syntactic Ghost: An Imperceptible General-purpose Backdoor Attacks on Pre-trained Language Models

    [https://arxiv.org/abs/2402.18945](https://arxiv.org/abs/2402.18945)

    论文提出了一种名为Syntactic Ghost的新方法，实现了对预训练语言模型进行无感知和通用的后门植入。

    

    预训练语言模型（PLMs）被发现容易受到后门攻击，可以将漏洞转移到各种下游任务中。然而，现有的PLM后门攻击采用明显的触发器，在手动对准的情况下进行，因此在效果、隐匿性和通用性方面无法同时满足期望目标。本文提出了一种新方法，实现了不可见和通用的后门植入，称为Syntactic Ghost（简称为synGhost）。具体来说，该方法敌意地使用具有不同预定义句法结构的毒害样本作为隐蔽触发器，然后将后门植入到预训练表示空间，而不会破坏原始知识。毒害样本的输出表示在特征空间中尽可能均匀地分布，通过对比学习形成广泛的后门。此外，在亮

    arXiv:2402.18945v1 Announce Type: cross  Abstract: Pre-trained language models (PLMs) have been found susceptible to backdoor attacks, which can transfer vulnerabilities to various downstream tasks. However, existing PLM backdoors are conducted with explicit triggers under the manually aligned, thus failing to satisfy expectation goals simultaneously in terms of effectiveness, stealthiness, and universality. In this paper, we propose a novel approach to achieve invisible and general backdoor implantation, called \textbf{Syntactic Ghost} (synGhost for short). Specifically, the method hostilely manipulates poisoned samples with different predefined syntactic structures as stealth triggers and then implants the backdoor to pre-trained representation space without disturbing the primitive knowledge. The output representations of poisoned samples are distributed as uniformly as possible in the feature space via contrastive learning, forming a wide range of backdoors. Additionally, in light 
    
[^3]: Permute-and-Flip：一种具有最佳鲁棒性和可加水印的LLMs解码器

    Permute-and-Flip: An optimally robust and watermarkable decoder for LLMs

    [https://arxiv.org/abs/2402.05864](https://arxiv.org/abs/2402.05864)

    提出了一种名为Permute-and-Flip（PF）解码器，其具有最佳的鲁棒性和质量-鲁棒性的 tradeoff，且比采样方法更好。还设计了一种针对PF解码器的水印方案，能够保持样本的分布不变，并实现任意低的假阳性率和高的召回率。实验证明PF解码器在困惑度方面明显优于朴素采样，为LLM解码提供了一种有希望的新方法。

    

    在本文中，我们提出了一种名为Permute-and-Flip（PF）解码器的新解码方法。它具有与标准采样解码器相似的鲁棒性特性，但在质量和鲁棒性的 tradeoff 上证明比采样方法更好，且永远不会差于任何其他解码器。同时，我们还设计了一种类似于Aaronson的Gumbel水印的加密水印方案，但是针对PF解码器而自然量身定制。该水印方案不改变样本的分布，同时允许任意低的假阳性率和高的召回率，只要生成的文本具有高熵。我们的实验证明，PF解码器（及其带有水印的对应物）在困惑度方面明显优于朴素采样（及其带有Gumbel水印的对应物），同时保持相同的鲁棒性（和可检测性），因此为LLM解码提供了一个有希望的新方法。代码可在https://github.com/XuandongZhao/pf-decoding找到。

    In this paper, we propose a new decoding method called Permute-and-Flip (PF) decoder. It enjoys robustness properties similar to the standard sampling decoder, but is provably up to 2x better in its quality-robustness tradeoff than sampling and never worse than any other decoder. We also design a cryptographic watermarking scheme analogous to Aaronson's Gumbel watermark, but naturally tailored for PF decoder. The watermarking scheme does not change the distribution to sample, while allowing arbitrarily low false positive rate and high recall whenever the generated text has high entropy. Our experiments show that the PF decoder (and its watermarked counterpart) significantly outperform(s) naive sampling (and it's Gumbel watermarked counterpart) in terms of perplexity, while retaining the same robustness (and detectability), hence making it a promising new approach for LLM decoding. The code is available at https://github.com/XuandongZhao/pf-decoding
    

