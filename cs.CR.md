# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Few-Shot Adversarial Prompt Learning on Vision-Language Models](https://arxiv.org/abs/2403.14774) | 本文提出了一个少样本对抗提示框架，在视觉-语言模型中通过有限数据调整输入序列，显著提升对抗鲁棒性，并通过端到端学习对抗性相关的文本监督。 |
| [^2] | [Learning to Poison Large Language Models During Instruction Tuning](https://arxiv.org/abs/2402.13459) | 通过设计新的数据注入攻击攻击LLMs，并提出一种梯度引导后门触发器学习方法，通过实验验证表明成功地破坏模型输出，仅改变1%的指导调优样本即可导致性能下降率达到约80％。 |
| [^3] | [Generative AI Security: Challenges and Countermeasures](https://arxiv.org/abs/2402.12617) | 生成式人工智能的安全挑战及对策研究。 |

# 详细

[^1]: 视觉-语言模型上的少样本对抗提示学习

    Few-Shot Adversarial Prompt Learning on Vision-Language Models

    [https://arxiv.org/abs/2403.14774](https://arxiv.org/abs/2403.14774)

    本文提出了一个少样本对抗提示框架，在视觉-语言模型中通过有限数据调整输入序列，显著提升对抗鲁棒性，并通过端到端学习对抗性相关的文本监督。

    

    深度神经网络对于微不可见的对抗性扰动的脆弱性已经引起了广泛关注。受到视觉-语言基础模型成功的启发，先前的努力通过将对抗性视觉特征与文本监督对齐来实现零样本对抗鲁棒性。但在实践中，由于包括重大适应成本、次优文本监督和未受控制的自然泛化能力在内的多个问题，它们仍然不尽人意。为了解决这些问题，本文提出了一个少样本对抗提示框架，通过有限的数据调整输入序列使得对抗鲁棒性得到显著提升。具体而言，我们通过提供对抗相关的文本监督，该监督是从对抗性示例中端到端学习的，来实现这一点。我们还提出了一个增强多模态特征一致性并鼓励不同

    arXiv:2403.14774v1 Announce Type: cross  Abstract: The vulnerability of deep neural networks to imperceptible adversarial perturbations has attracted widespread attention. Inspired by the success of vision-language foundation models, previous efforts achieved zero-shot adversarial robustness by aligning adversarial visual features with text supervision. However, in practice, they are still unsatisfactory due to several issues, including heavy adaptation cost, suboptimal text supervision, and uncontrolled natural generalization capacity. In this paper, to address these issues, we propose a few-shot adversarial prompt framework where adapting input sequences with limited data makes significant adversarial robustness improvement. Specifically, we achieve this by providing adversarially correlated text supervision that is end-to-end learned from adversarial examples. We also propose a novel training objective that enhances the consistency of multi-modal features while encourages differenti
    
[^2]: 学习在指导调优期间操纵大型语言模型

    Learning to Poison Large Language Models During Instruction Tuning

    [https://arxiv.org/abs/2402.13459](https://arxiv.org/abs/2402.13459)

    通过设计新的数据注入攻击攻击LLMs，并提出一种梯度引导后门触发器学习方法，通过实验验证表明成功地破坏模型输出，仅改变1%的指导调优样本即可导致性能下降率达到约80％。

    

    大型语言模型（LLMs）的出现标志着语言处理和推理能力方面的重大突破。虽然它们取得了显著进展，但LLMs面临着数据注入攻击的漏洞，其中对手将后门触发器插入训练数据，以操纵输出以进行恶意行为。本研究通过设计一种新的数据注入攻击，旨在利用指导调优过程，进一步识别LLMs中的额外安全风险。我们提出了一种新颖的梯度引导后门触发器学习方法，以有效识别敌对触发器，确保对传统防御手段的规避，同时保持内容的完整性。通过对各种LLMs和任务的实验验证，我们的策略表明在破坏模型输出方面取得了很高的成功率；仅对4,000个指导调优样本中的1％进行注入就导致性能降低率（PDR）约为80％。我们的工作高

    arXiv:2402.13459v1 Announce Type: cross  Abstract: The advent of Large Language Models (LLMs) has marked significant achievements in language processing and reasoning capabilities. Despite their advancements, LLMs face vulnerabilities to data poisoning attacks, where adversaries insert backdoor triggers into training data to manipulate outputs for malicious purposes. This work further identifies additional security risks in LLMs by designing a new data poisoning attack tailored to exploit the instruction tuning process. We propose a novel gradient-guided backdoor trigger learning approach to identify adversarial triggers efficiently, ensuring an evasion of detection by conventional defenses while maintaining content integrity. Through experimental validation across various LLMs and tasks, our strategy demonstrates a high success rate in compromising model outputs; poisoning only 1\% of 4,000 instruction tuning samples leads to a Performance Drop Rate (PDR) of around 80\%. Our work high
    
[^3]: 生成式人工智能安全：挑战与对策

    Generative AI Security: Challenges and Countermeasures

    [https://arxiv.org/abs/2402.12617](https://arxiv.org/abs/2402.12617)

    生成式人工智能的安全挑战及对策研究。

    

    arXiv:2402.12617v1 公告类型：跨领域 摘要：生成式人工智能在许多行业的不断扩展引发了人们的兴奋和增加的关注。本文深入探讨了生成式人工智能所带来的独特安全挑战，并概述了管理这些风险的潜在研究方向。

    arXiv:2402.12617v1 Announce Type: cross  Abstract: Generative AI's expanding footprint across numerous industries has led to both excitement and increased scrutiny. This paper delves into the unique security challenges posed by Generative AI, and outlines potential research directions for managing these risks.
    

