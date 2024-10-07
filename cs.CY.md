# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Can Language Models Recognize Convincing Arguments?](https://arxiv.org/abs/2404.00750) | 大语言模型不仅能够在识别和区分强势和弱势论点方面表现良好，还可以根据用户的信念和人口特征预测其立场，并确定论点对个人的吸引力。 |
| [^2] | [GreenLLaMA: A Framework for Detoxification with Explanations](https://arxiv.org/abs/2402.15951) | GreenLLaMA是一种全面的端到端解毒框架，通过跨平台语料库训练出的模型优于当前最先进的模型。 |
| [^3] | [SMILE: Single-turn to Multi-turn Inclusive Language Expansion via ChatGPT for Mental Health Support.](http://arxiv.org/abs/2305.00450) | 本研究提出了SMILE方法，使用ChatGPT将公共单轮对话扩展为多轮对话，生成了大规模、多样化、接近真实生活的多轮心理健康支持对话语料库，可用于训练和评估专门的对话系统。 |

# 详细

[^1]: 大语言模型能识别令人信服的论点吗？

    Can Language Models Recognize Convincing Arguments?

    [https://arxiv.org/abs/2404.00750](https://arxiv.org/abs/2404.00750)

    大语言模型不仅能够在识别和区分强势和弱势论点方面表现良好，还可以根据用户的信念和人口特征预测其立场，并确定论点对个人的吸引力。

    

    大型语言模型（LLMs）的显著且不断增强的能力引发了人们对它们可能被滥用用来创造个性化、令人信服的虚假信息和宣传的担忧。为了深入了解LLMs的说服能力，而又不直接与人类进行实验，我们提出研究它们在检测令人信服的论点任务上的表现。我们通过添加辩论、投票和用户特征来扩展了Durmus和Cardie（2018）的数据集，并提出了衡量LLMs能力的任务，包括（1）区分强势和弱势论点，（2）基于信念和人口特征预测立场，以及（3）根据个人特征确定对一个论点的吸引力。我们发现LLMs在这些任务中表现与人类不相上下，并且结合不同LLMs的预测可以获得显著的性能提升，甚至超过人类的表现。随文附带发布的数据和代码。

    arXiv:2404.00750v1 Announce Type: new  Abstract: The remarkable and ever-increasing capabilities of Large Language Models (LLMs) have raised concerns about their potential misuse for creating personalized, convincing misinformation and propaganda. To gain insights into LLMs' persuasive capabilities without directly engaging in experimentation with humans, we propose studying their performance on the related task of detecting convincing arguments. We extend a dataset by Durmus & Cardie (2018) with debates, votes, and user traits and propose tasks measuring LLMs' ability to (1) distinguish between strong and weak arguments, (2) predict stances based on beliefs and demographic characteristics, and (3) determine the appeal of an argument to an individual based on their traits. We show that LLMs perform on par with humans in these tasks and that combining predictions from different LLMs yields significant performance gains, even surpassing human performance. The data and code released with 
    
[^2]: GreenLLaMA: 一种带有解释的解毒框架

    GreenLLaMA: A Framework for Detoxification with Explanations

    [https://arxiv.org/abs/2402.15951](https://arxiv.org/abs/2402.15951)

    GreenLLaMA是一种全面的端到端解毒框架，通过跨平台语料库训练出的模型优于当前最先进的模型。

    

    先前关于解毒的研究工作分散在某种程度上，因为它们并没有涵盖到真实场景中所需的所有解毒方面。值得注意的是，先前的研究将开发解毒模型的任务局限在仅见过的平台子集上，没有探讨模型在未知平台上的表现如何。此外，这些工作没有解决不可解毒性这一现象，即毒性文本无法在不改变含义的情况下进行解毒。我们提出了GreenLLaMA，这是第一个全面的端到端解毒框架，旨在减轻上述限制。我们首先介绍了一个跨平台伪并行语料库，应用多步数据处理和生成策略利用ChatGPT。然后，我们使用跨平台语料库训练一套解毒模型。我们展示了我们的解毒模型优于使用人工注释的最先进模型的表现。

    arXiv:2402.15951v1 Announce Type: cross  Abstract: Prior works on detoxification are scattered in the sense that they do not cover all aspects of detoxification needed in a real-world scenario. Notably, prior works restrict the task of developing detoxification models to only a seen subset of platforms, leaving the question of how the models would perform on unseen platforms unexplored. Additionally, these works do not address non-detoxifiability, a phenomenon whereby the toxic text cannot be detoxified without altering the meaning. We propose GreenLLaMA, the first comprehensive end-to-end detoxification framework, which attempts to alleviate the aforementioned limitations. We first introduce a cross-platform pseudo-parallel corpus applying multi-step data processing and generation strategies leveraging ChatGPT. We then train a suite of detoxification models with our cross-platform corpus. We show that our detoxification models outperform the SoTA model trained with human-annotated par
    
[^3]: SMILE：利用ChatGPT实现单轮到多轮包容性语言扩展的心理健康支持

    SMILE: Single-turn to Multi-turn Inclusive Language Expansion via ChatGPT for Mental Health Support. (arXiv:2305.00450v1 [cs.CL])

    [http://arxiv.org/abs/2305.00450](http://arxiv.org/abs/2305.00450)

    本研究提出了SMILE方法，使用ChatGPT将公共单轮对话扩展为多轮对话，生成了大规模、多样化、接近真实生活的多轮心理健康支持对话语料库，可用于训练和评估专门的对话系统。

    

    开发专门的对话系统以提供心理健康支持已成为越来越多的研究关注点。然而，由于个人信息的敏感性以及所需的时间和成本，获取大规模的真实多轮心理健康支持对话存在困难。为了解决这些问题，我们引入了SMILE方法，一种使用ChatGPT将公共单轮对话扩展为多轮对话的包容性语言扩展技术。我们首先进行了初步的探索性研究，验证了SMILE方法的有效性。此外，我们对使用和未使用SMILE方法生成的数据集进行了全面系统的对比分析，证明SMILE方法可以产生大规模、多样化、接近真实生活的多轮心理健康支持对话语料库，包括对话主题、词汇和语义特征。最后，我们使用收集的语料库来训练和评估专门的心理健康支持对话系统。

    There has been an increasing research interest in developing specialized dialogue systems that can offer mental health support. However, gathering large-scale and real-life multi-turn conversations for mental health support poses challenges due to the sensitivity of personal information, as well as the time and cost involved. To address these issues, we introduce the SMILE approach, an inclusive language expansion technique that employs ChatGPT to extend public single-turn dialogues into multi-turn ones. Our research first presents a preliminary exploratory study that validates the effectiveness of the SMILE approach. Furthermore, we conduct a comprehensive and systematic contrastive analysis of datasets generated with and without the SMILE approach, demonstrating that the SMILE method results in a large-scale, diverse, and close-to-real-life multi-turn mental health support conversation corpus, including dialog topics, lexical and semantic features. Finally, we use the collected corpu
    

