# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Controlled Reevaluation of Coreference Resolution Models](https://arxiv.org/abs/2404.00727) | 基于对预训练语言模型大小的控制，我们发现基于编码器的核指代解析模型在准确性和推理速度方面优于更近期的基于解码器的模型，而在基于编码器的模型中，最老的模型在跨域文本体裁中表现最佳。 |
| [^2] | [Syntactic Ghost: An Imperceptible General-purpose Backdoor Attacks on Pre-trained Language Models](https://arxiv.org/abs/2402.18945) | 论文提出了一种名为Syntactic Ghost的新方法，实现了对预训练语言模型进行无感知和通用的后门植入。 |

# 详细

[^1]: 核指代解析模型的受控重新评估

    A Controlled Reevaluation of Coreference Resolution Models

    [https://arxiv.org/abs/2404.00727](https://arxiv.org/abs/2404.00727)

    基于对预训练语言模型大小的控制，我们发现基于编码器的核指代解析模型在准确性和推理速度方面优于更近期的基于解码器的模型，而在基于编码器的模型中，最老的模型在跨域文本体裁中表现最佳。

    

    所有最先进的核指代解析（CR）模型都涉及微调预训练语言模型。一个CR模型优于另一个的出色性能是由语言模型选择还是其他因素（如特定于任务的架构）造成的，由于缺乏标准化的实验设置，这是很难或不可能确定的。为了解决这种模糊性，我们系统评估了五个CR模型，并控制了一些设计决策，包括每个模型使用的预训练语言模型。当控制语言模型大小时，基于编码器的CR模型在准确性和推理速度方面优于更近期的基于解码器的模型。令人惊讶的是，在基于编码器的CR模型中，较近期的模型并不总是更准确，而我们测试的最老的CR模型在跨域文本体裁中表现最佳。我们得出结论，控制语言模型的选择可以减少大部分，但并非全部，

    arXiv:2404.00727v1 Announce Type: new  Abstract: All state-of-the-art coreference resolution (CR) models involve finetuning a pretrained language model. Whether the superior performance of one CR model over another is due to the choice of language model or other factors, such as the task-specific architecture, is difficult or impossible to determine due to lack of a standardized experimental setup. To resolve this ambiguity, we systematically evaluate five CR models and control for certain design decisions including the pretrained language model used by each. When controlling for language model size, encoder-based CR models outperform more recent decoder-based models in terms of both accuracy and inference speed. Surprisingly, among encoder-based CR models, more recent models are not always more accurate, and the oldest CR model that we test generalizes the best to out-of-domain textual genres. We conclude that controlling for the choice of language model reduces most, but not all, of 
    
[^2]: Syntactic Ghost：一种对预训练语言模型进行的无感知通用后门攻击

    Syntactic Ghost: An Imperceptible General-purpose Backdoor Attacks on Pre-trained Language Models

    [https://arxiv.org/abs/2402.18945](https://arxiv.org/abs/2402.18945)

    论文提出了一种名为Syntactic Ghost的新方法，实现了对预训练语言模型进行无感知和通用的后门植入。

    

    预训练语言模型（PLMs）被发现容易受到后门攻击，可以将漏洞转移到各种下游任务中。然而，现有的PLM后门攻击采用明显的触发器，在手动对准的情况下进行，因此在效果、隐匿性和通用性方面无法同时满足期望目标。本文提出了一种新方法，实现了不可见和通用的后门植入，称为Syntactic Ghost（简称为synGhost）。具体来说，该方法敌意地使用具有不同预定义句法结构的毒害样本作为隐蔽触发器，然后将后门植入到预训练表示空间，而不会破坏原始知识。毒害样本的输出表示在特征空间中尽可能均匀地分布，通过对比学习形成广泛的后门。此外，在亮

    arXiv:2402.18945v1 Announce Type: cross  Abstract: Pre-trained language models (PLMs) have been found susceptible to backdoor attacks, which can transfer vulnerabilities to various downstream tasks. However, existing PLM backdoors are conducted with explicit triggers under the manually aligned, thus failing to satisfy expectation goals simultaneously in terms of effectiveness, stealthiness, and universality. In this paper, we propose a novel approach to achieve invisible and general backdoor implantation, called \textbf{Syntactic Ghost} (synGhost for short). Specifically, the method hostilely manipulates poisoned samples with different predefined syntactic structures as stealth triggers and then implants the backdoor to pre-trained representation space without disturbing the primitive knowledge. The output representations of poisoned samples are distributed as uniformly as possible in the feature space via contrastive learning, forming a wide range of backdoors. Additionally, in light 
    

