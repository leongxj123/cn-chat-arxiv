# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deciphering the Interplay between Local Differential Privacy, Average Bayesian Privacy, and Maximum Bayesian Privacy](https://arxiv.org/abs/2403.16591) | 论文探讨了本地差分隐私、贝叶斯隐私及其之间的相互关系，揭示了关于效用-隐私权衡的新见解，并提出了一个框架来突出攻击和防御策略的相互作用和效果。 |
| [^2] | [URLBERT:A Contrastive and Adversarial Pre-trained Model for URL Classification](https://arxiv.org/abs/2402.11495) | URLBERT是第一个专门针对URL分类或检测任务的预训练模型，引入了自监督对比学习和虚拟对抗训练两种新颖的预训练任务，以加强模型对URL结构的理解和提高从URL中提取语义特征的鲁棒性。 |
| [^3] | [Comprehensive Assessment of Jailbreak Attacks Against LLMs](https://arxiv.org/abs/2402.05668) | 对大型语言模型（LLMs）的越狱攻击进行了全面的评估，揭示了一种绕过安全措施的不稳定漏洞。本研究是首次对多种越狱攻击方法进行大规模测量，实验证明优化的越狱提示能够持续达到最高的攻击成功率。 |

# 详细

[^1]: 揭示本地差分隐私、平均贝叶斯隐私和最大贝叶斯隐私之间的相互作用

    Deciphering the Interplay between Local Differential Privacy, Average Bayesian Privacy, and Maximum Bayesian Privacy

    [https://arxiv.org/abs/2403.16591](https://arxiv.org/abs/2403.16591)

    论文探讨了本地差分隐私、贝叶斯隐私及其之间的相互关系，揭示了关于效用-隐私权衡的新见解，并提出了一个框架来突出攻击和防御策略的相互作用和效果。

    

    机器学习的迅速发展导致了隐私定义的多样化，由于对隐私构成的威胁，包括本地差分隐私（LDP）的概念。虽然被广泛接受并在许多领域中被利用，但这种传统的隐私测量方法仍然存在一定限制，从无法防止推断披露到缺乏对对手背景知识的考虑。在这项全面研究中，我们引入贝叶斯隐私并深入探讨本地差分隐私和其贝叶斯对应物之间错综复杂的关系，揭示了关于效用-隐私权衡的新见解。我们引入了一个框架，概括了攻击和防御策略，突出它们之间的相互作用和效果。我们的理论贡献基于平均贝叶斯隐私（ABP）和最大贝叶斯隐私之间的严格定义和关系。

    arXiv:2403.16591v1 Announce Type: cross  Abstract: The swift evolution of machine learning has led to emergence of various definitions of privacy due to the threats it poses to privacy, including the concept of local differential privacy (LDP). Although widely embraced and utilized across numerous domains, this conventional approach to measure privacy still exhibits certain limitations, spanning from failure to prevent inferential disclosure to lack of consideration for the adversary's background knowledge. In this comprehensive study, we introduce Bayesian privacy and delve into the intricate relationship between local differential privacy and its Bayesian counterparts, unveiling novel insights into utility-privacy trade-offs. We introduce a framework that encapsulates both attack and defense strategies, highlighting their interplay and effectiveness. Our theoretical contributions are anchored in the rigorous definitions and relationships between Average Bayesian Privacy (ABP) and Max
    
[^2]: URLBERT：一种用于URL分类的对比和对抗预训练模型

    URLBERT:A Contrastive and Adversarial Pre-trained Model for URL Classification

    [https://arxiv.org/abs/2402.11495](https://arxiv.org/abs/2402.11495)

    URLBERT是第一个专门针对URL分类或检测任务的预训练模型，引入了自监督对比学习和虚拟对抗训练两种新颖的预训练任务，以加强模型对URL结构的理解和提高从URL中提取语义特征的鲁棒性。

    

    arXiv：2402.11495v1 发表类型：跨领域摘要：URL在理解和分类网络内容方面发挥着至关重要的作用，特别是在与安全控制和在线推荐相关的任务中。尽管预训练模型目前在各个领域占据主导地位，但URL分析领域仍缺乏专门的预训练模型。为填补这一空白，本文介绍了URLBERT，这是第一个应用于各种URL分类或检测任务的预训练表示学习模型。我们首先在数十亿个URL的语料库上训练了一个URL标记器，以解决URL数据的标记化问题。此外，我们提出了两种新颖的预训练任务：（1）自监督对比学习任务，通过区分相同URL的不同变体来增强模型对URL结构的理解和对类别差异的捕捉；（2）虚拟对抗训练，旨在提高模型从URL中提取语义特征的鲁棒性。最后，我们提出了

    arXiv:2402.11495v1 Announce Type: cross  Abstract: URLs play a crucial role in understanding and categorizing web content, particularly in tasks related to security control and online recommendations. While pre-trained models are currently dominating various fields, the domain of URL analysis still lacks specialized pre-trained models. To address this gap, this paper introduces URLBERT, the first pre-trained representation learning model applied to a variety of URL classification or detection tasks. We first train a URL tokenizer on a corpus of billions of URLs to address URL data tokenization. Additionally, we propose two novel pre-training tasks: (1) self-supervised contrastive learning tasks, which strengthen the model's understanding of URL structure and the capture of category differences by distinguishing different variants of the same URL; (2) virtual adversarial training, aimed at improving the model's robustness in extracting semantic features from URLs. Finally, our proposed 
    
[^3]: 对LLMs的越狱攻击的综合评估

    Comprehensive Assessment of Jailbreak Attacks Against LLMs

    [https://arxiv.org/abs/2402.05668](https://arxiv.org/abs/2402.05668)

    对大型语言模型（LLMs）的越狱攻击进行了全面的评估，揭示了一种绕过安全措施的不稳定漏洞。本研究是首次对多种越狱攻击方法进行大规模测量，实验证明优化的越狱提示能够持续达到最高的攻击成功率。

    

    对大型语言模型（LLMs）的滥用引起了广泛关注。为了解决这个问题，已经采取了安全措施以确保LLMs符合社会伦理。然而，最近的研究发现了一种绕过LLMs安全措施的不稳定漏洞，被称为越狱攻击。通过应用技术，如角色扮演场景、对抗性样本或对安全目标的微妙破坏作为提示，LLMs可以产生不适当甚至有害的回应。虽然研究人员已经研究了几种越狱攻击的类别，但他们都是孤立地进行的。为了填补这个空白，我们提出了对各种越狱攻击方法的首次大规模测量。我们集中在来自四个类别的13种尖端越狱方法、16种违规类别的160个问题以及六种流行的LLMs上。我们广泛的实验结果表明，优化的越狱提示始终能够达到最高的攻击成功率，并表现出...

    Misuse of the Large Language Models (LLMs) has raised widespread concern. To address this issue, safeguards have been taken to ensure that LLMs align with social ethics. However, recent findings have revealed an unsettling vulnerability bypassing the safeguards of LLMs, known as jailbreak attacks. By applying techniques, such as employing role-playing scenarios, adversarial examples, or subtle subversion of safety objectives as a prompt, LLMs can produce an inappropriate or even harmful response. While researchers have studied several categories of jailbreak attacks, they have done so in isolation. To fill this gap, we present the first large-scale measurement of various jailbreak attack methods. We concentrate on 13 cutting-edge jailbreak methods from four categories, 160 questions from 16 violation categories, and six popular LLMs. Our extensive experimental results demonstrate that the optimized jailbreak prompts consistently achieve the highest attack success rates, as well as exhi
    

