# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Image Hijacking: Adversarial Images can Control Generative Models at Runtime.](http://arxiv.org/abs/2309.00236) | 本研究发现对抗性图像能够在运行时控制生成模型，并提出了通用的方法来创建图像劫持。通过研究三种攻击类型，我们发现这些攻击对最新的视觉语言模型具有高达90％以上的成功率。该研究引发了对基础模型安全性的严重担忧。 |

# 详细

[^1]: 图像劫持：对抗性图像能在运行时控制生成模型

    Image Hijacking: Adversarial Images can Control Generative Models at Runtime. (arXiv:2309.00236v1 [cs.LG])

    [http://arxiv.org/abs/2309.00236](http://arxiv.org/abs/2309.00236)

    本研究发现对抗性图像能够在运行时控制生成模型，并提出了通用的方法来创建图像劫持。通过研究三种攻击类型，我们发现这些攻击对最新的视觉语言模型具有高达90％以上的成功率。该研究引发了对基础模型安全性的严重担忧。

    

    基础模型是否能够免受恶意行为者的攻击？本文研究了视觉语言模型（VLM）的图像输入。我们发现了图像劫持，即能够在运行时控制生成模型的对抗性图像。我们引入了一种名为“行为匹配”的通用方法来创建图像劫持，并用它来探索三种类型的攻击：具体字符串攻击可以生成任意被攻击者选择的输出；泄露上下文攻击可以将上下文窗口中的信息泄露到输出中；越狱攻击可以绕过模型的安全训练。我们对基于CLIP和LLaMA-2的最新VLM模型LLaVA-2进行了这些攻击的研究，并发现我们所有的攻击类型成功率均在90％以上。而且，我们的攻击是自动化的，只需要对图像进行小的扰动。这些发现对基础模型的安全性提出了严重的担忧。如果图像劫持与CIFAR-10中的对抗性样本一样难以防御，那么可能需要很多年才能找到解决方案。

    Are foundation models secure from malicious actors? In this work, we focus on the image input to a vision-language model (VLM). We discover image hijacks, adversarial images that control generative models at runtime. We introduce Behavior Matching, a general method for creating image hijacks, and we use it to explore three types of attacks. Specific string attacks generate arbitrary output of the adversary's choosing. Leak context attacks leak information from the context window into the output. Jailbreak attacks circumvent a model's safety training. We study these attacks against LLaVA-2, a state-of-the-art VLM based on CLIP and LLaMA-2, and find that all our attack types have above a 90\% success rate. Moreover, our attacks are automated and require only small image perturbations. These findings raise serious concerns about the security of foundation models. If image hijacks are as difficult to defend against as adversarial examples in CIFAR-10, then it might be many years before a s
    

