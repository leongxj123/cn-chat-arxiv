# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [One Graph Model for Cross-domain Dynamic Link Prediction](https://arxiv.org/abs/2402.02168) | DyExpert是一种用于跨域链接预测的动态图模型，通过明确建模历史演化过程并结合链接预测，它可以学习特定下游图的演化模式，并在各个领域上取得了最先进的性能。 |

# 详细

[^1]: 跨域动态链接预测的一种图模型

    One Graph Model for Cross-domain Dynamic Link Prediction

    [https://arxiv.org/abs/2402.02168](https://arxiv.org/abs/2402.02168)

    DyExpert是一种用于跨域链接预测的动态图模型，通过明确建模历史演化过程并结合链接预测，它可以学习特定下游图的演化模式，并在各个领域上取得了最先进的性能。

    

    本研究提出了DyExpert，一种用于跨域链接预测的动态图模型。它可以明确地建模历史演化过程，学习特定下游图的演化模式，并进而进行特定模式的链接预测。DyExpert采用了解码器优化的transformer，并通过结合演化建模和链接预测的“条件链接生成”实现了高效的并行训练和推断。DyExpert在包含6百万个动态边的广泛动态图上进行训练。在八个未训练的图上进行了大量实验，结果显示DyExpert在跨域链接预测中取得了最先进的性能。与相同设置下的先进基准相比，DyExpert在八个图上的平均精确度提高了11.40％。更令人印象深刻的是，在六个未训练的图上，它超过了八个先进基线的全监督性能。

    This work proposes DyExpert, a dynamic graph model for cross-domain link prediction. It can explicitly model historical evolving processes to learn the evolution pattern of a specific downstream graph and subsequently make pattern-specific link predictions. DyExpert adopts a decode-only transformer and is capable of efficiently parallel training and inference by \textit{conditioned link generation} that integrates both evolution modeling and link prediction. DyExpert is trained by extensive dynamic graphs across diverse domains, comprising 6M dynamic edges. Extensive experiments on eight untrained graphs demonstrate that DyExpert achieves state-of-the-art performance in cross-domain link prediction. Compared to the advanced baseline under the same setting, DyExpert achieves an average of 11.40% improvement Average Precision across eight graphs. More impressive, it surpasses the fully supervised performance of 8 advanced baselines on 6 untrained graphs.
    

