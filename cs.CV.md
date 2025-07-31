# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CLoRA: A Contrastive Approach to Compose Multiple LoRA Models](https://arxiv.org/abs/2403.19776) | CLoRA提出了一种对比方法，用于组合多个LoRA模型，解决了将不同概念LoRA模型无缝混合到一个图像中的挑战。 |

# 详细

[^1]: CLoRA: 一种对比方法来组合多个 LoRA 模型

    CLoRA: A Contrastive Approach to Compose Multiple LoRA Models

    [https://arxiv.org/abs/2403.19776](https://arxiv.org/abs/2403.19776)

    CLoRA提出了一种对比方法，用于组合多个LoRA模型，解决了将不同概念LoRA模型无缝混合到一个图像中的挑战。

    

    低秩调整（LoRA）已经成为图像生成领域中一种强大且受欢迎的技术，提供了一种高效的方式来调整和改进预训练的深度学习模型，而无需全面地重新训练。通过使用预训练的 LoRA 模型，例如代表特定猫和特定狗的模型，我们的目标是生成一个图像，该图像真实地体现了 LoRA 所定义的两种动物。然而，无缝地混合多个概念 LoRA 模型以捕获一个图像中的各种概念的任务被证明是一个重大挑战。常见方法往往表现不佳，主要是因为不同 LoRA 模型内的注意机制重叠，导致一个概念可能被完全忽略（例如漏掉了狗），或者概念被错误地组合在一起（例如生成两只猫的图像而不是一只猫和一只狗）。为了克服这一挑战，

    arXiv:2403.19776v1 Announce Type: cross  Abstract: Low-Rank Adaptations (LoRAs) have emerged as a powerful and popular technique in the field of image generation, offering a highly effective way to adapt and refine pre-trained deep learning models for specific tasks without the need for comprehensive retraining. By employing pre-trained LoRA models, such as those representing a specific cat and a particular dog, the objective is to generate an image that faithfully embodies both animals as defined by the LoRAs. However, the task of seamlessly blending multiple concept LoRAs to capture a variety of concepts in one image proves to be a significant challenge. Common approaches often fall short, primarily because the attention mechanisms within different LoRA models overlap, leading to scenarios where one concept may be completely ignored (e.g., omitting the dog) or where concepts are incorrectly combined (e.g., producing an image of two cats instead of one cat and one dog). To overcome th
    

