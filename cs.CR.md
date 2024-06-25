# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Jailbreaking is Best Solved by Definition](https://arxiv.org/abs/2403.14725) | 语言模型中"越狱"攻击的关键是通过定义好的不安全响应来进行防御，而不是依赖于执行策略。 |
| [^2] | [Private Benchmarking to Prevent Contamination and Improve Comparative Evaluation of LLMs](https://arxiv.org/abs/2403.00393) | 本论文提出了私人基准设定方法，通过保持测试数据的私密性，使模型在不揭露测试数据的情况下进行评估，解决了当前基准设定中存在数据污染的问题。 |
| [^3] | [Watermark Stealing in Large Language Models](https://arxiv.org/abs/2402.19361) | LLM水印技术可能存在水印窃取漏洞，我们提出了自动WS算法并展示了攻击者可以在不到50美元的成本下通过欺骗和擦除攻击破解之前认为安全的最先进方案，成功率超过80%。 |
| [^4] | [Pandora's White-Box: Increased Training Data Leakage in Open LLMs](https://arxiv.org/abs/2402.17012) | 本文对开源大型语言模型（LLMs）进行了隐私攻击研究，提出了首个能同时实现高真正率和低误分类率的预训练LLMs会员推理攻击（MIAs），以及展示了在自然环境中可以从微调LLM中提取超过50%的微调数据集。 |

# 详细

[^1]: Jailbreaking的最佳解决方案是通过定义

    Jailbreaking is Best Solved by Definition

    [https://arxiv.org/abs/2403.14725](https://arxiv.org/abs/2403.14725)

    语言模型中"越狱"攻击的关键是通过定义好的不安全响应来进行防御，而不是依赖于执行策略。

    

    语言模型上"越狱"攻击的增多引发了大量防御工作，旨在防止产生不良回应。在这项工作中，我们批判性地审视了防御管道的两个阶段：（i）定义何为不安全输出，和（ii）通过输入处理或微调等方法来执行该定义。我们严重怀疑现有的执行机制的有效性，通过展示它们即使对于简单的不安全输出定义--包含单词"purple"的输出也无法防御。相比之下，对输出进行后处理对于这样的定义是完全健壮的。基于我们的结果，我们提出我们的观点，即在防御越狱攻击中真正的挑战在于得到一个良好的不安全响应定义：没有良好的定义，任何执行策略都无法成功，但有了良好的定义，输出处理已经作为一个强大的基线。

    arXiv:2403.14725v1 Announce Type: cross  Abstract: The rise of "jailbreak" attacks on language models has led to a flurry of defenses aimed at preventing the output of undesirable responses. In this work, we critically examine the two stages of the defense pipeline: (i) the definition of what constitutes unsafe outputs, and (ii) the enforcement of the definition via methods such as input processing or fine-tuning. We cast severe doubt on the efficacy of existing enforcement mechanisms by showing that they fail to defend even for a simple definition of unsafe outputs--outputs that contain the word "purple". In contrast, post-processing outputs is perfectly robust for such a definition. Drawing on our results, we present our position that the real challenge in defending jailbreaks lies in obtaining a good definition of unsafe responses: without a good definition, no enforcement strategy can succeed, but with a good definition, output processing already serves as a robust baseline albeit 
    
[^2]: 防止污染和提高LLM比较评估的私人基准设定

    Private Benchmarking to Prevent Contamination and Improve Comparative Evaluation of LLMs

    [https://arxiv.org/abs/2403.00393](https://arxiv.org/abs/2403.00393)

    本论文提出了私人基准设定方法，通过保持测试数据的私密性，使模型在不揭露测试数据的情况下进行评估，解决了当前基准设定中存在数据污染的问题。

    

    基准设定是评估LLM的事实标准，因为它速度快、可复制且成本低廉。然而，最近的研究指出，今天大多数开源基准设定已经被污染或泄露到LLM中，这意味着LLM在预训练和/或微调期间可以访问测试数据。这对迄今为止进行的基准研究的有效性以及未来使用基准进行评估提出了严重关切。为了解决这个问题，我们提出了Private Benchmarking，这是一个方案，其中测试数据集保持私密，模型在不向模型透露测试数据的情况下进行评估。我们描述了各种场景（取决于对模型所有者或数据集所有者的信任），并提出了使用私人基准设定避免数据污染的解决方案。对于需要保护模型权重的情况，我们描述了来自机密计算和密码学的解决方案。

    arXiv:2403.00393v1 Announce Type: cross  Abstract: Benchmarking is the de-facto standard for evaluating LLMs, due to its speed, replicability and low cost. However, recent work has pointed out that the majority of the open source benchmarks available today have been contaminated or leaked into LLMs, meaning that LLMs have access to test data during pretraining and/or fine-tuning. This raises serious concerns about the validity of benchmarking studies conducted so far and the future of evaluation using benchmarks. To solve this problem, we propose Private Benchmarking, a solution where test datasets are kept private and models are evaluated without revealing the test data to the model. We describe various scenarios (depending on the trust placed on model owners or dataset owners), and present solutions to avoid data contamination using private benchmarking. For scenarios where the model weights need to be kept private, we describe solutions from confidential computing and cryptography t
    
[^3]: 大型语言模型中的水印窃取

    Watermark Stealing in Large Language Models

    [https://arxiv.org/abs/2402.19361](https://arxiv.org/abs/2402.19361)

    LLM水印技术可能存在水印窃取漏洞，我们提出了自动WS算法并展示了攻击者可以在不到50美元的成本下通过欺骗和擦除攻击破解之前认为安全的最先进方案，成功率超过80%。

    

    LLM水印技术作为一种检测AI生成内容的有效方式，受到了关注。然而，我们在这项研究中争辩称当前方案可能已经可以部署，我们认为水印窃取（WS）是这些方案的一个根本性漏洞。我们展示了通过查询带有水印的LLM的API来近似逆向水印，从而实现实用的欺骗攻击，同时大幅增加了之前未被注意到的擦除攻击。我们是第一个提出自动WS算法并将其用于在现实环境中进行欺骗和擦除的全面研究。我们展示了仅需不到50美元的成本，攻击者就能够欺骗并擦除之前被认为是安全的最先进方案，平均成功率超过80%。我们的研究挑战了关于LLM水印技术的常见信念，强调了更加健壮方案的必要性。

    arXiv:2402.19361v1 Announce Type: cross  Abstract: LLM watermarking has attracted attention as a promising way to detect AI-generated content, with some works suggesting that current schemes may already be fit for deployment. In this work we dispute this claim, identifying watermark stealing (WS) as a fundamental vulnerability of these schemes. We show that querying the API of the watermarked LLM to approximately reverse-engineer a watermark enables practical spoofing attacks, as suggested in prior work, but also greatly boosts scrubbing attacks, which was previously unnoticed. We are the first to propose an automated WS algorithm and use it in the first comprehensive study of spoofing and scrubbing in realistic settings. We show that for under $50 an attacker can both spoof and scrub state-of-the-art schemes previously considered safe, with average success rate of over 80%. Our findings challenge common beliefs about LLM watermarking, stressing the need for more robust schemes. We mak
    
[^4]: Pandora's White-Box：开放LLMs中训练数据泄漏的增加

    Pandora's White-Box: Increased Training Data Leakage in Open LLMs

    [https://arxiv.org/abs/2402.17012](https://arxiv.org/abs/2402.17012)

    本文对开源大型语言模型（LLMs）进行了隐私攻击研究，提出了首个能同时实现高真正率和低误分类率的预训练LLMs会员推理攻击（MIAs），以及展示了在自然环境中可以从微调LLM中提取超过50%的微调数据集。

    

    在本文中，我们对开源的大型语言模型（LLMs）遭受的隐私攻击进行了系统研究，其中对手可以访问模型权重、梯度或损失，试图利用它们来了解底层训练数据。我们的主要结果是针对预训练LLMs的第一个会员推理攻击（MIAs），能够同时实现高TPR和低FPR，并展示了在自然环境中可以从微调LLM中提取超过50%的微调数据集。我们考虑了对底层模型的不同访问程度、语言模型的定制化以及攻击者可以使用的资源。在预训练设置中，我们提出了三种新的白盒MIAs：基于梯度范数的攻击、监督神经网络分类器和单步损失比攻击。所有这些都优于现有的黑盒基线，并且我们的.....

    arXiv:2402.17012v1 Announce Type: cross  Abstract: In this paper we undertake a systematic study of privacy attacks against open source Large Language Models (LLMs), where an adversary has access to either the model weights, gradients, or losses, and tries to exploit them to learn something about the underlying training data. Our headline results are the first membership inference attacks (MIAs) against pre-trained LLMs that are able to simultaneously achieve high TPRs and low FPRs, and a pipeline showing that over $50\%$ (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, customization of the language model, and resources available to the attacker. In the pre-trained setting, we propose three new white-box MIAs: an attack based on the gradient norm, a supervised neural network classifier, and a single step loss ratio attack. All outperform existing black-box baselines, and our supervi
    

