# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Open Sesame! Universal Black Box Jailbreaking of Large Language Models.](http://arxiv.org/abs/2309.01446) | 本文提出了一种使用遗传算法的新颖方法，可以在无法访问模型架构和参数的情况下操纵大规模语言模型 (LLMs)。通过优化通用对抗提示与用户查询结合，可以扰乱被攻击模型的对齐，导致意外和潜在有害的输出。该方法可以揭示模型的局限性和漏洞，为负责任的AI开发提供了一种诊断工具。 |

# 详细

[^1]: 开门吧！大规模语言模型的通用黑盒破解

    Open Sesame! Universal Black Box Jailbreaking of Large Language Models. (arXiv:2309.01446v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2309.01446](http://arxiv.org/abs/2309.01446)

    本文提出了一种使用遗传算法的新颖方法，可以在无法访问模型架构和参数的情况下操纵大规模语言模型 (LLMs)。通过优化通用对抗提示与用户查询结合，可以扰乱被攻击模型的对齐，导致意外和潜在有害的输出。该方法可以揭示模型的局限性和漏洞，为负责任的AI开发提供了一种诊断工具。

    

    大规模语言模型（LLMs）旨在提供有帮助和安全的回复，通常依赖于对齐技术与用户意图和社会指南保持一致。然而，这种对齐可能会被恶意行为者利用，以用于意想不到的目的。在本文中，我们引入了一种新颖的方法，利用遗传算法（GA）在模型架构和参数不可访问时操纵LLMs。GA攻击通过优化通用对抗提示与用户查询结合，扰乱被攻击模型的对齐，导致意外和潜在有害的输出。我们的新颖方法通过揭示模型的局限性和漏洞，系统地揭示了其响应与预期行为不符的情况。通过广泛的实验，我们证明了我们的技术的有效性，从而为关于负责任的AI开发的讨论提供了一种诊断工具。

    Large language models (LLMs), designed to provide helpful and safe responses, often rely on alignment techniques to align with user intent and social guidelines. Unfortunately, this alignment can be exploited by malicious actors seeking to manipulate an LLM's outputs for unintended purposes. In this paper we introduce a novel approach that employs a genetic algorithm (GA) to manipulate LLMs when model architecture and parameters are inaccessible. The GA attack works by optimizing a universal adversarial prompt that -- when combined with a user's query -- disrupts the attacked model's alignment, resulting in unintended and potentially harmful outputs. Our novel approach systematically reveals a model's limitations and vulnerabilities by uncovering instances where its responses deviate from expected behavior. Through extensive experiments we demonstrate the efficacy of our technique, thus contributing to the ongoing discussion on responsible AI development by providing a diagnostic tool 
    

