# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [General Phrase Debiaser: Debiasing Masked Language Models at a Multi-Token Level.](http://arxiv.org/abs/2311.13892) | 本文提出了一种名为“通用短语去偏器”的自动多标记去偏管道，能够有效减轻掩码语言模型中的短语级别偏见，并在标准数据集和指标上取得了最新成果。 |
| [^2] | [Amortizing intractable inference in large language models.](http://arxiv.org/abs/2310.04363) | 本论文提出了一种使用摊销的贝叶斯推理从难以处理的后验分布中进行抽样的方法，并利用生成流网络来实现大规模语言模型的微调，从而解决了在这些模型中处理推理问题的限制。 |

# 详细

[^1]: 通用短语去偏器：在多标记级别上消除掩码语言模型中的偏见

    General Phrase Debiaser: Debiasing Masked Language Models at a Multi-Token Level. (arXiv:2311.13892v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2311.13892](http://arxiv.org/abs/2311.13892)

    本文提出了一种名为“通用短语去偏器”的自动多标记去偏管道，能够有效减轻掩码语言模型中的短语级别偏见，并在标准数据集和指标上取得了最新成果。

    

    预训练语言模型所揭示的社会偏见和不受欢迎的刻板印象正在成为其应用的障碍。与针对词级别的众多去偏方法相比，对于短语级别的偏见关注相对较少，限制了学科领域去偏的性能。在本文中，我们提出了一种名为“通用短语去偏器”的自动多标记去偏管道，能够减轻掩码语言模型中的短语级别偏见。具体而言，我们的方法包括一个“短语过滤阶段”，从维基百科页面中生成刻板印象的短语，以及一个“模型去偏阶段”，可以在多标记级别上去偏模型以应对短语上的偏见挑战。后者寻找触发模型偏见的提示，然后将其用于去偏。标准数据集和评估指标上的最新成果表明，我们的方法可以显著减少职业和加强性别偏见。

    The social biases and unwelcome stereotypes revealed by pretrained language models are becoming obstacles to their application. Compared to numerous debiasing methods targeting word level, there has been relatively less attention on biases present at phrase level, limiting the performance of debiasing in discipline domains. In this paper, we propose an automatic multi-token debiasing pipeline called \textbf{General Phrase Debiaser}, which is capable of mitigating phrase-level biases in masked language models. Specifically, our method consists of a \textit{phrase filter stage} that generates stereotypical phrases from Wikipedia pages as well as a \textit{model debias stage} that can debias models at the multi-token level to tackle bias challenges on phrases. The latter searches for prompts that trigger model's bias, and then uses them for debiasing. State-of-the-art results on standard datasets and metrics show that our approach can significantly reduce gender biases on both career and 
    
[^2]: 在大规模语言模型中摊销难以处理的推理问题

    Amortizing intractable inference in large language models. (arXiv:2310.04363v1 [cs.LG])

    [http://arxiv.org/abs/2310.04363](http://arxiv.org/abs/2310.04363)

    本论文提出了一种使用摊销的贝叶斯推理从难以处理的后验分布中进行抽样的方法，并利用生成流网络来实现大规模语言模型的微调，从而解决了在这些模型中处理推理问题的限制。

    

    自回归的大规模语言模型通过下一个词条件分布来压缩其训练数据中的知识，这限制了对该知识的可处理查询仅限于从头到尾的自回归抽样。然而，许多感兴趣的任务，包括序列延续、填充和其他形式的受约束生成，都涉及从难以处理的后验分布中进行抽样。我们通过使用摊销的贝叶斯推理从这些难以处理的后验分布中进行抽样来解决这个限制。这种摊销通过通过寻求多样性的强化学习算法 - 生成流网络 (GFlowNets) 来微调 LLMs 实现。我们凭经验证明，LLM微调的这种分布匹配范式可以作为最大似然训练和奖励最大化策略优化的有效替代方法。作为一个重要应用，我们将思维链推理解释为潜变量建模问题，并证明了...

    Autoregressive large language models (LLMs) compress knowledge from their training data through next-token conditional distributions. This limits tractable querying of this knowledge to start-to-end autoregressive sampling. However, many tasks of interest -- including sequence continuation, infilling, and other forms of constrained generation -- involve sampling from intractable posterior distributions. We address this limitation by using amortized Bayesian inference to sample from these intractable posteriors. Such amortization is algorithmically achieved by fine-tuning LLMs via diversity-seeking reinforcement learning algorithms: generative flow networks (GFlowNets). We empirically demonstrate that this distribution-matching paradigm of LLM fine-tuning can serve as an effective alternative to maximum-likelihood training and reward-maximizing policy optimization. As an important application, we interpret chain-of-thought reasoning as a latent variable modeling problem and demonstrate 
    

