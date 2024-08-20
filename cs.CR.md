# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Topic-based Watermarks for LLM-Generated Text](https://arxiv.org/abs/2404.02138) | 提出了一种新的基于主题的水印算法，旨在解决当前水印方案的局限性，为区分LLM生成的文本和人类生成的文本提供了新的思路。 |
| [^2] | [Symbiotic Game and Foundation Models for Cyber Deception Operations in Strategic Cyber Warfare](https://arxiv.org/abs/2403.10570) | 博弈论模型和基础模型在分析、设计和实施网络欺骗策略中发挥关键作用，为提升主动和自动化网络防御机制提供了新思路。 |
| [^3] | [Proving membership in LLM pretraining data via data watermarks](https://arxiv.org/abs/2402.10892) | 使用数据水印在LLM预训练中检测版权持有人作品的方法，可以进行合理检测且提供误检率保证，研究了水印设计对假设检验能力的影响以及在模型和数据集缩放下的检测强度变化。 |
| [^4] | [Bergeron: Combating Adversarial Attacks through a Conscience-Based Alignment Framework](https://arxiv.org/abs/2312.00029) | Bergeron提出了一个基于良知的对齆框架，能够提高大型语言模型对抗攻击的鲁棒性，无需额外参数微调。 |

# 详细

[^1]: 基于主题的LLM生成文本的水印

    Topic-based Watermarks for LLM-Generated Text

    [https://arxiv.org/abs/2404.02138](https://arxiv.org/abs/2404.02138)

    提出了一种新的基于主题的水印算法，旨在解决当前水印方案的局限性，为区分LLM生成的文本和人类生成的文本提供了新的思路。

    

    大型语言模型（LLMs）的最新进展导致了生成的文本输出与人类生成的文本相似度难以分辨。水印算法是潜在工具，通过在LLM生成的输出中嵌入可检测的签名，可以区分LLM生成的文本和人类生成的文本。然而，当前的水印方案在已知攻击下缺乏健壮性。此外，考虑到LLM每天生成数万个文本输出，水印算法需要记忆每个输出才能让检测正常工作，这是不切实际的。本文针对当前水印方案的局限性，提出了针对LLMs的“基于主题的水印算法”概念。

    arXiv:2404.02138v1 Announce Type: cross  Abstract: Recent advancements of large language models (LLMs) have resulted in indistinguishable text outputs comparable to human-generated text. Watermarking algorithms are potential tools that offer a way to differentiate between LLM- and human-generated text by embedding detectable signatures within LLM-generated output. However, current watermarking schemes lack robustness against known attacks against watermarking algorithms. In addition, they are impractical considering an LLM generates tens of thousands of text outputs per day and the watermarking algorithm needs to memorize each output it generates for the detection to work. In this work, focusing on the limitations of current watermarking schemes, we propose the concept of a "topic-based watermarking algorithm" for LLMs. The proposed algorithm determines how to generate tokens for the watermarked LLM output based on extracted topics of an input prompt or the output of a non-watermarked 
    
[^2]: 战略网络战中的生物共生游戏和基础模型

    Symbiotic Game and Foundation Models for Cyber Deception Operations in Strategic Cyber Warfare

    [https://arxiv.org/abs/2403.10570](https://arxiv.org/abs/2403.10570)

    博弈论模型和基础模型在分析、设计和实施网络欺骗策略中发挥关键作用，为提升主动和自动化网络防御机制提供了新思路。

    

    面临着网络战战术的快速演变、情报不对称性增加和黑客工具的日益易得，我们正面临前所未有的网络战。在这种背景下，网络欺骗作为我们防御策略中的关键组成部分崭露头角，旨在应对日益复杂的攻击。本章旨在强调博弈论模型和基础模型（FMs）在分析、设计和实施网络欺骗策略中的关键作用。博弈模型（GMs）作为一个基础框架，用于建模多样的对抗性交互，使我们能够包容对抗性知识和领域特定的见解。同时，基础模型作为创建针对特定应用的定制机器学习模型的构建块。通过利用博弈模型和基础模型之间的协同效应，我们可以通过不仅保护我们的网络免受攻击，而且提高主动和自动化网络防御机制。

    arXiv:2403.10570v1 Announce Type: cross  Abstract: We are currently facing unprecedented cyber warfare with the rapid evolution of tactics, increasing asymmetry of intelligence, and the growing accessibility of hacking tools. In this landscape, cyber deception emerges as a critical component of our defense strategy against increasingly sophisticated attacks. This chapter aims to highlight the pivotal role of game-theoretic models and foundation models (FMs) in analyzing, designing, and implementing cyber deception tactics. Game models (GMs) serve as a foundational framework for modeling diverse adversarial interactions, allowing us to encapsulate both adversarial knowledge and domain-specific insights. Meanwhile, FMs serve as the building blocks for creating tailored machine learning models suited to given applications. By leveraging the synergy between GMs and FMs, we can advance proactive and automated cyber defense mechanisms by not only securing our networks against attacks but als
    
[^3]: 通过数据水印证明LLM预训练数据的成员资格

    Proving membership in LLM pretraining data via data watermarks

    [https://arxiv.org/abs/2402.10892](https://arxiv.org/abs/2402.10892)

    使用数据水印在LLM预训练中检测版权持有人作品的方法，可以进行合理检测且提供误检率保证，研究了水印设计对假设检验能力的影响以及在模型和数据集缩放下的检测强度变化。

    

    检测版权持有人的作品是否在LLM预训练中使用是一个重要问题，本文提出使用数据水印实现基于黑盒模型访问的合理检测，前提是版权持有人在公开发布之前贡献了多个训练文档并对其进行了水印处理。通过应用随机采样的数据水印，检测可以被构造为假设检验，从而提供对误检率的保证。研究了两种水印：一种插入随机序列，另一种随机用Unicode类似字符替换字符。首先展示了水印设计的三个方面--水印长度、复制次数和干扰--如何影响假设检验的能力。接着研究了水印在模型和数据集缩放下的检测强度如何变化：增加数据集大小会降低水印的强度，水印...

    arXiv:2402.10892v1 Announce Type: cross  Abstract: Detecting whether copyright holders' works were used in LLM pretraining is poised to be an important problem. This work proposes using data watermarks to enable principled detection with only black-box model access, provided that the rightholder contributed multiple training documents and watermarked them before public release. By applying a randomly sampled data watermark, detection can be framed as hypothesis testing, which provides guarantees on the false detection rate. We study two watermarks: one that inserts random sequences, and another that randomly substitutes characters with Unicode lookalikes. We first show how three aspects of watermark design -- watermark length, number of duplications, and interference -- affect the power of the hypothesis test. Next, we study how a watermark's detection strength changes under model and dataset scaling: while increasing the dataset size decreases the strength of the watermark, watermarks
    
[^4]: 通过基于良知的对准框架抵御对抗性攻击

    Bergeron: Combating Adversarial Attacks through a Conscience-Based Alignment Framework

    [https://arxiv.org/abs/2312.00029](https://arxiv.org/abs/2312.00029)

    Bergeron提出了一个基于良知的对齆框架，能够提高大型语言模型对抗攻击的鲁棒性，无需额外参数微调。

    

    近年来，随着越来越强大的大型语言模型（LLMs）的引入，人工智能对齐的研究取得了可观的进展。不幸的是，现代对齐方法仍然无法完全防止在模型被蓄意攻击时产生有害应对。为了帮助缓解这一问题，我们引入了Bergeron：一个旨在提高LLMs对抗攻击鲁棒性的框架，无需进行额外的参数微调。Bergeron分为两个层次；次要LLM模拟受保护的主要LLM的良知。该框架在监视输出以检测任何有害内容的同时，更好地保护主要模型免受入侵攻击。实证分析表明，使用Bergeron来补充现有对齐训练的模型

    arXiv:2312.00029v2 Announce Type: replace-cross  Abstract: Research into AI alignment has grown considerably since the recent introduction of increasingly capable Large Language Models (LLMs). Unfortunately, modern methods of alignment still fail to fully prevent harmful responses when models are deliberately attacked. These attacks can trick seemingly aligned models into giving manufacturing instructions for dangerous materials, inciting violence, or recommending other immoral acts. To help mitigate this issue, we introduce Bergeron: a framework designed to improve the robustness of LLMs against attacks without any additional parameter fine-tuning. Bergeron is organized into two tiers; with a secondary LLM emulating the conscience of a protected, primary LLM. This framework better safeguards the primary model against incoming attacks while monitoring its output for any harmful content. Empirical analysis shows that, by using Bergeron to complement models with existing alignment traini
    

