# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TransAxx: Efficient Transformers with Approximate Computing](https://arxiv.org/abs/2402.07545) | TransAxx是一个基于PyTorch库的框架，可以支持近似计算，并通过对Vision Transformer模型进行近似感知微调，来提高在低功耗设备上的计算效率。 |

# 详细

[^1]: TransAxx：具有近似计算能力的高效Transformer模型

    TransAxx: Efficient Transformers with Approximate Computing

    [https://arxiv.org/abs/2402.07545](https://arxiv.org/abs/2402.07545)

    TransAxx是一个基于PyTorch库的框架，可以支持近似计算，并通过对Vision Transformer模型进行近似感知微调，来提高在低功耗设备上的计算效率。

    

    最近，基于Transformer架构引入的Vision Transformer (ViT)模型已经展现出很大的竞争力，并且往往成为卷积神经网络(CNNs)的一种流行替代方案。然而，这些模型高计算需求限制了它们在低功耗设备上的实际应用。当前最先进的方法采用近似乘法器来解决DNN加速器高计算需求的问题，但之前的研究并没有探索其在ViT模型上的应用。在这项工作中，我们提出了TransAxx，这是一个基于流行的PyTorch库的框架，它能够快速支持近似算术，以无缝地评估近似计算对于DNN (如ViT模型)的影响。使用TransAxx，我们分析了Transformer模型在ImageNet数据集上对近似乘法的敏感性，并进行了近似感知的微调以恢复准确性。此外，我们提出了一种生成近似加法的方法。

    Vision Transformer (ViT) models which were recently introduced by the transformer architecture have shown to be very competitive and often become a popular alternative to Convolutional Neural Networks (CNNs). However, the high computational requirements of these models limit their practical applicability especially on low-power devices. Current state-of-the-art employs approximate multipliers to address the highly increased compute demands of DNN accelerators but no prior research has explored their use on ViT models. In this work we propose TransAxx, a framework based on the popular PyTorch library that enables fast inherent support for approximate arithmetic to seamlessly evaluate the impact of approximate computing on DNNs such as ViT models. Using TransAxx we analyze the sensitivity of transformer models on the ImageNet dataset to approximate multiplications and perform approximate-aware finetuning to regain accuracy. Furthermore, we propose a methodology to generate approximate ac
    

