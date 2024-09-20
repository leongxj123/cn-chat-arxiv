# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention.](http://arxiv.org/abs/2303.16199) | 本文提出了一种基于适应提示和零初始化注意力机制的轻量级语言模型调整方法，可高效微调LLaMA为指令跟随模型，具有比Alpaca更短的微调时间并具有近似的响应质量。 |

# 详细

[^1]: LLaMA-Adapter: 零初始化注意力下的语言模型精细调整的高效方法

    LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention. (arXiv:2303.16199v1 [cs.CV])

    [http://arxiv.org/abs/2303.16199](http://arxiv.org/abs/2303.16199)

    本文提出了一种基于适应提示和零初始化注意力机制的轻量级语言模型调整方法，可高效微调LLaMA为指令跟随模型，具有比Alpaca更短的微调时间并具有近似的响应质量。

    

    本文提出了LLaMA-Adapter这一轻量级适应方法，用于将LLaMA高效地微调为一个指令跟随模型。利用52K个自我指导示范，LLaMA-Adapter仅在冻结的LLaMA 7B模型上引入了1.2M个可学习参数，并且在8个A100 GPU上仅耗时不到一个小时进行微调。具体而言，我们采用一组可学习的适应提示，并在较高的变压器层中将它们预置于输入文本令牌之前。然后，提出了一种零初始化注意力机制和零门控机制，该机制可以自适应地将新的指令提示注入LLaMA，并有效地保留了其预先训练的知识。通过高效训练，LLaMA-Adapter能够产生高质量的响应，与完全微调的7B参数的Alpaca相似。此外，我们的方法还可以简单地扩展到多模态输入，例如图像，用于图像相关的LLaMA，在ScienceQA上实现了更强的推理能力。我们在https://github.com/ZrrSkywalker/LLaMA-Adapt发布了我们的代码。

    We present LLaMA-Adapter, a lightweight adaption method to efficiently fine-tune LLaMA into an instruction-following model. Using 52K self-instruct demonstrations, LLaMA-Adapter only introduces 1.2M learnable parameters upon the frozen LLaMA 7B model, and costs less than one hour for fine-tuning on 8 A100 GPUs. Specifically, we adopt a set of learnable adaption prompts, and prepend them to the input text tokens at higher transformer layers. Then, a zero-init attention mechanism with zero gating is proposed, which adaptively injects the new instructional cues into LLaMA, while effectively preserves its pre-trained knowledge. With efficient training, LLaMA-Adapter generates high-quality responses, comparable to Alpaca with fully fine-tuned 7B parameters. Furthermore, our approach can be simply extended to multi-modal input, e.g., images, for image-conditioned LLaMA, which achieves superior reasoning capacity on ScienceQA. We release our code at https://github.com/ZrrSkywalker/LLaMA-Adapt
    

