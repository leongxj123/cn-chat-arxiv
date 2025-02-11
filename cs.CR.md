# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Topic-based Watermarks for LLM-Generated Text](https://arxiv.org/abs/2404.02138) | 提出了一种新的基于主题的水印算法，旨在解决当前水印方案的局限性，为区分LLM生成的文本和人类生成的文本提供了新的思路。 |
| [^2] | [LOCALINTEL: Generating Organizational Threat Intelligence from Global and Local Cyber Knowledge.](http://arxiv.org/abs/2401.10036) | LOCALINTEL是一个自动化的知识上下文化系统，利用大型语言模型的能力，从全球和本地知识数据库中自动生成组织的威胁情报。 |

# 详细

[^1]: 基于主题的LLM生成文本的水印

    Topic-based Watermarks for LLM-Generated Text

    [https://arxiv.org/abs/2404.02138](https://arxiv.org/abs/2404.02138)

    提出了一种新的基于主题的水印算法，旨在解决当前水印方案的局限性，为区分LLM生成的文本和人类生成的文本提供了新的思路。

    

    大型语言模型（LLMs）的最新进展导致了生成的文本输出与人类生成的文本相似度难以分辨。水印算法是潜在工具，通过在LLM生成的输出中嵌入可检测的签名，可以区分LLM生成的文本和人类生成的文本。然而，当前的水印方案在已知攻击下缺乏健壮性。此外，考虑到LLM每天生成数万个文本输出，水印算法需要记忆每个输出才能让检测正常工作，这是不切实际的。本文针对当前水印方案的局限性，提出了针对LLMs的“基于主题的水印算法”概念。

    arXiv:2404.02138v1 Announce Type: cross  Abstract: Recent advancements of large language models (LLMs) have resulted in indistinguishable text outputs comparable to human-generated text. Watermarking algorithms are potential tools that offer a way to differentiate between LLM- and human-generated text by embedding detectable signatures within LLM-generated output. However, current watermarking schemes lack robustness against known attacks against watermarking algorithms. In addition, they are impractical considering an LLM generates tens of thousands of text outputs per day and the watermarking algorithm needs to memorize each output it generates for the detection to work. In this work, focusing on the limitations of current watermarking schemes, we propose the concept of a "topic-based watermarking algorithm" for LLMs. The proposed algorithm determines how to generate tokens for the watermarked LLM output based on extracted topics of an input prompt or the output of a non-watermarked 
    
[^2]: LOCALINTEL：从全球和本地网络知识生成组织威胁情报

    LOCALINTEL: Generating Organizational Threat Intelligence from Global and Local Cyber Knowledge. (arXiv:2401.10036v1 [cs.CR])

    [http://arxiv.org/abs/2401.10036](http://arxiv.org/abs/2401.10036)

    LOCALINTEL是一个自动化的知识上下文化系统，利用大型语言模型的能力，从全球和本地知识数据库中自动生成组织的威胁情报。

    

    安全操作中心（SoC）分析师从公开访问的全球威胁数据库中收集威胁报告，并手动自定义以适应特定组织的需求。这些分析师还依赖于内部存储库，作为组织的私有本地知识数据库。可信的网络情报、关键操作细节和相关组织信息都存储在这些本地知识数据库中。分析师利用这些全球和本地知识数据库从事一项繁重的任务，手动创建组织独特的威胁响应和缓解策略。最近，大型语言模型（LLMs）已经展示了高效处理大规模多样化知识源的能力。我们利用这种能力来处理全球和本地知识数据库，自动化生成组织特定的威胁情报。在这项工作中，我们提出了LOCALINTEL，这是一个新颖的自动化知识上下文化系统，可以从全球和本地知识数据库中生成组织的威胁情报。

    Security Operations Center (SoC) analysts gather threat reports from openly accessible global threat databases and customize them manually to suit a particular organization's needs. These analysts also depend on internal repositories, which act as private local knowledge database for an organization. Credible cyber intelligence, critical operational details, and relevant organizational information are all stored in these local knowledge databases. Analysts undertake a labor intensive task utilizing these global and local knowledge databases to manually create organization's unique threat response and mitigation strategies. Recently, Large Language Models (LLMs) have shown the capability to efficiently process large diverse knowledge sources. We leverage this ability to process global and local knowledge databases to automate the generation of organization-specific threat intelligence.  In this work, we present LOCALINTEL, a novel automated knowledge contextualization system that, upon 
    

