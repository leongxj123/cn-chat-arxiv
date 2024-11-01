# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Studious Bob Fight Back Against Jailbreaking via Prompt Adversarial Tuning](https://arxiv.org/abs/2402.06255) | 本文提出了一种名为Prompt Adversarial Tuning (PAT)的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中，实现对大型语言模型（LLMs）的越狱行为的防御。实验证明该方法在保护LLMs免受产生有害信息的影响方面效果显著。 |
| [^2] | [Tree of Attacks: Jailbreaking Black-Box LLMs Automatically](https://arxiv.org/abs/2312.02119) | 提出了一种名为Tree of Attacks with Pruning (TAP)的自动化方法，用于生成只需要对目标大型语言模型进行黑盒访问的越狱方法，并通过思维树推理和修剪生成准确的越狱提示。 |
| [^3] | [Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning.](http://arxiv.org/abs/2401.10862) | 本文研究了剪枝对齐的LLMs的保护措施，发现剪枝LLM参数可以显著增强其抵抗“越狱”提示攻击的能力，并且对其他LLM行为也可能有更普遍的效果。同时，引入了一个有害任务数据集，证明剪枝有助于集中注意力在与任务相关的标记上。突出的聊天模型表现出很高的易感性。 |
| [^4] | [Invisible Image Watermarks Are Provably Removable Using Generative AI.](http://arxiv.org/abs/2306.01953) | 该论文证明了使用生成式AI可以清除隐形图像水印，提出了一种家族化再生攻击方法。通过形式化证明和实证结果，论文展示了所有隐形水印容易受到攻击，并针对一种具有弹性的水印RivaGAN，再生攻击可以去除93-99%的水印。 |
| [^5] | [Pre-trained transformer for adversarial purification.](http://arxiv.org/abs/2306.01762) | 本文提出了一个快速防御对抗性攻击的方案RaPiD（Rapid Plug-in Defender），通过预训练的Transformer微调来提纯对抗样本，使其逼近清洁数据分布，实验结果表明，在有限数据情况下，该方法优于最先进的方法。 |

# 详细

[^1]: 进取的鲍勃通过提示对抗调整抵制越狱行为

    Studious Bob Fight Back Against Jailbreaking via Prompt Adversarial Tuning

    [https://arxiv.org/abs/2402.06255](https://arxiv.org/abs/2402.06255)

    本文提出了一种名为Prompt Adversarial Tuning (PAT)的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中，实现对大型语言模型（LLMs）的越狱行为的防御。实验证明该方法在保护LLMs免受产生有害信息的影响方面效果显著。

    

    尽管大型语言模型（LLM）在各种应用中取得了巨大的成功，但它们也容易受到特定提示的影响，从而绕过内置的安全措施并提供危险或非法内容，这种现象被称为越狱行为。为了保护LLMs免受产生有害信息的影响，提出了各种防御策略，其中大多数集中在内容过滤或模型的对抗训练方面。在本文中，我们提出了一种名为Prompt Adversarial Tuning（PAT）的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中来实现我们的防御策略。我们设计了一个类似对抗训练的训练过程，以实现我们的优化目标，交替更新攻击和防御控制机制。据我们所知，我们是第一个从提示调整的角度实施防御的人。一旦应用，我们的方法几乎不会影响LLMs的操作效率。实验表明我们的方法在抵御越狱行为方面具有良好的效果。

    Although Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to certain prompts that can induce them to bypass built-in safety measures and provide dangerous or illegal content, a phenomenon known as jailbreak. To protect LLMs from producing harmful information, various defense strategies are proposed, with most focusing on content filtering or adversarial training of models. In this paper, we propose an approach named Prompt Adversarial Tuning (PAT) to train a defense control mechanism, which is then embedded as a prefix to user prompts to implement our defense strategy. We design a training process similar to adversarial training to achieve our optimized goal, alternating between updating attack and defense controls. To our knowledge, we are the first to implement defense from the perspective of prompt tuning. Once employed, our method will hardly impact the operational efficiency of LLMs. Experiments show that our method i
    
[^2]: 攻击树：自动破解黑盒大型语言模型

    Tree of Attacks: Jailbreaking Black-Box LLMs Automatically

    [https://arxiv.org/abs/2312.02119](https://arxiv.org/abs/2312.02119)

    提出了一种名为Tree of Attacks with Pruning (TAP)的自动化方法，用于生成只需要对目标大型语言模型进行黑盒访问的越狱方法，并通过思维树推理和修剪生成准确的越狱提示。

    

    大型语言模型(LLMs)展示了多功能性，但仍在生成有害、带偏见和有毒内容，这一点由人为设计的越狱行为的普遍存在得以证明。在这项工作中，我们提出了一种名为Tree of Attacks with Pruning (TAP)的自动化方法，用于生成越狱，仅需要对目标LLM进行黑盒访问。TAP利用LLM来通过思维树推理迭代地优化候选（攻击）提示，直到生成的提示之一越狱目标。关键在于，在将提示发送给目标之前，TAP对其进行评估并移除可能不会导致越狱的提示。使用思维树推理使TAP能够在大量提示的搜索空间中导航，而修剪则减少了发送给目标的总查询数量。在实证评估中，我们观察到TAP生成的提示越狱了超过80%的最先进LLMs（包括GPT4和GPT4-Turbo）。

    arXiv:2312.02119v2 Announce Type: replace-cross  Abstract: While Large Language Models (LLMs) display versatile functionality, they continue to generate harmful, biased, and toxic content, as demonstrated by the prevalence of human-designed jailbreaks. In this work, we present Tree of Attacks with Pruning (TAP), an automated method for generating jailbreaks that only requires black-box access to the target LLM. TAP utilizes an LLM to iteratively refine candidate (attack) prompts using tree-of-thought reasoning until one of the generated prompts jailbreaks the target. Crucially, before sending prompts to the target, TAP assesses them and prunes the ones unlikely to result in jailbreaks. Using tree-of-thought reasoning allows TAP to navigate a large search space of prompts and pruning reduces the total number of queries sent to the target. In empirical evaluations, we observe that TAP generates prompts that jailbreak state-of-the-art LLMs (including GPT4 and GPT4-Turbo) for more than 80%
    
[^3]: 基于剪枝的保护: 在不进行微调的情况下增加对齐的LLMs的越狱抵抗力

    Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning. (arXiv:2401.10862v1 [cs.LG])

    [http://arxiv.org/abs/2401.10862](http://arxiv.org/abs/2401.10862)

    本文研究了剪枝对齐的LLMs的保护措施，发现剪枝LLM参数可以显著增强其抵抗“越狱”提示攻击的能力，并且对其他LLM行为也可能有更普遍的效果。同时，引入了一个有害任务数据集，证明剪枝有助于集中注意力在与任务相关的标记上。突出的聊天模型表现出很高的易感性。

    

    大型语言模型（LLMs）容易受到“越狱”提示的攻击，这种攻击可以诱使这些模型生成有害和违法内容。本文表明，剪枝LLM参数多达20％可以显著增加它们对此类攻击的抵抗力，而无需额外训练并且不损害其在标准基准测试中的性能。有趣的是，我们发现剪枝后观察到的增强安全性与模型的初始安全训练水平相关，这暗示剪枝的效果可能更普遍，也可能适用于超出安全性范畴的其他LLM行为。另外，我们还介绍了一个包含五个类别、插入到十个不同越狱提示中的225个有害任务的精选数据集，表明剪枝有助于LLMs集中注意力在越狱提示中与任务相关的标记上。最后，我们的实验揭示了突出的聊天模型（如LLaMA-2 Chat，Vicuna和Mistral Instruct）具有很高的易感性。

    Large Language Models (LLMs) are vulnerable to `Jailbreaking' prompts, a type of attack that can coax these models into generating harmful and illegal content. In this paper, we show that pruning up to 20% of LLM parameters markedly increases their resistance to such attacks without additional training and without sacrificing their performance in standard benchmarks. Intriguingly, we discovered that the enhanced safety observed post-pruning correlates to the initial safety training level of the model, hinting that the effect of pruning could be more general and may hold for other LLM behaviors beyond safety. Additionally, we introduce a curated dataset of 225 harmful tasks across five categories, inserted into ten different Jailbreaking prompts, showing that pruning aids LLMs in concentrating attention on task-relevant tokens in jailbreaking prompts. Lastly, our experiments reveal that the prominent chat models, such as LLaMA-2 Chat, Vicuna, and Mistral Instruct exhibit high susceptibi
    
[^4]: 通过生成式AI，证明了隐形图像水印是可清除的

    Invisible Image Watermarks Are Provably Removable Using Generative AI. (arXiv:2306.01953v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2306.01953](http://arxiv.org/abs/2306.01953)

    该论文证明了使用生成式AI可以清除隐形图像水印，提出了一种家族化再生攻击方法。通过形式化证明和实证结果，论文展示了所有隐形水印容易受到攻击，并针对一种具有弹性的水印RivaGAN，再生攻击可以去除93-99%的水印。

    

    隐形水印通过嵌入只有权利拥有者可以检测到的隐藏信息来保护图像的版权。它们还防止人们滥用由AI模型生成的图像。我们提出了一种家族化再生攻击来清除这些隐形水印。所提出的攻击方法首先向图像添加随机噪声来破坏水印，然后重建图像。这种方法灵活，可以与许多现有的图像降噪算法和预训练的生成模型（如扩散模型）实例化。通过形式化证明和实证结果，我们证明了所有隐形水印都容易受到所提出的攻击。对于一个特别有弹性的水印RivaGAN，再生攻击可以去除93-99%的隐形水印，而基线攻击只能去除不超过3%。然而，如果我们不要求带水印的图像与原始图像相同，保持图像语义相似的水印可能是一种替代方案。

    Invisible watermarks safeguard images' copyright by embedding hidden messages only detectable by owners. They also prevent people from misusing images, especially those generated by AI models. We propose a family of regeneration attacks to remove these invisible watermarks. The proposed attack method first adds random noise to an image to destroy the watermark and then reconstructs the image. This approach is flexible and can be instantiated with many existing image-denoising algorithms and pre-trained generative models such as diffusion models. Through formal proofs and empirical results, we show that all invisible watermarks are vulnerable to the proposed attack. For a particularly resilient watermark, RivaGAN, regeneration attacks remove 93-99% of the invisible watermarks while the baseline attacks remove no more than 3%. However, if we do not require the watermarked image to look the same as the original one, watermarks that keep the image semantically similar can be an alternative
    
[^5]: 预训练Transformer用于对抗性样本提纯

    Pre-trained transformer for adversarial purification. (arXiv:2306.01762v1 [cs.CR])

    [http://arxiv.org/abs/2306.01762](http://arxiv.org/abs/2306.01762)

    本文提出了一个快速防御对抗性攻击的方案RaPiD（Rapid Plug-in Defender），通过预训练的Transformer微调来提纯对抗样本，使其逼近清洁数据分布，实验结果表明，在有限数据情况下，该方法优于最先进的方法。

    

    随着越来越多的深度神经网络被部署为各种日常服务，它们的可靠性至关重要。深度神经网络容易受到对抗性攻击的影响，其中逃避攻击是最普遍的一种。最近的研究通常通过对抗训练或利用大量清洁数据的知识来增强其健壮性。然而，在实际应用中，重新训练和部署模型需要大量的计算资源，对在线服务造成重大损失。此外，当检测到某种攻击的对抗性例子时，服务提供者只能获得有限的对抗性样本，而大量的清洁数据可能无法获取。针对这些问题，我们提出了一种新的方案，名为RaPiD（Rapid Plug-in Defender），旨在快速防御具有少量干净和对抗性示例限制的原始服务模型的某种攻击。受到预训练模型提供转移学习良好初始化的通用趋势的启发，我们建议通过微调预先训练的Transformer来提纯对抗性样本。预训练的Transformer作为正则化器，鼓励提纯后的对抗性样本接近清晰数据的分布。实验结果表明，RaPiD在防御各种具有限数据的攻击方面优于最先进的方法。

    With more and more deep neural networks being deployed as various daily services, their reliability is essential. It's frightening that deep neural networks are vulnerable and sensitive to adversarial attacks, the most common one of which for the services is evasion-based. Recent works usually strengthen the robustness by adversarial training or leveraging the knowledge of an amount of clean data. However, in practical terms, retraining and redeploying the model need a large computational budget, leading to heavy losses to the online service. In addition, when adversarial examples of a certain attack are detected, only limited adversarial examples are available for the service provider, while much clean data may not be accessible. Given the mentioned problems, we propose a new scenario, RaPiD (Rapid Plug-in Defender), which is to rapidly defend against a certain attack for the frozen original service model with limitations of few clean and adversarial examples. Motivated by the general
    

