# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Permute-and-Flip: An optimally robust and watermarkable decoder for LLMs](https://arxiv.org/abs/2402.05864) | 提出了一种名为Permute-and-Flip（PF）解码器，其具有最佳的鲁棒性和质量-鲁棒性的 tradeoff，且比采样方法更好。还设计了一种针对PF解码器的水印方案，能够保持样本的分布不变，并实现任意低的假阳性率和高的召回率。实验证明PF解码器在困惑度方面明显优于朴素采样，为LLM解码提供了一种有希望的新方法。 |
| [^2] | [Model Stealing Attack against Multi-Exit Networks.](http://arxiv.org/abs/2305.13584) | 该论文介绍了第一个能同时窃取多出口网络模型函数和输出策略的攻击方法，并使用贝叶斯变点检测和性能损失、策略损失指导替代模型的训练。开发了一种新的输出策略搜索方法。 |

# 详细

[^1]: Permute-and-Flip：一种具有最佳鲁棒性和可加水印的LLMs解码器

    Permute-and-Flip: An optimally robust and watermarkable decoder for LLMs

    [https://arxiv.org/abs/2402.05864](https://arxiv.org/abs/2402.05864)

    提出了一种名为Permute-and-Flip（PF）解码器，其具有最佳的鲁棒性和质量-鲁棒性的 tradeoff，且比采样方法更好。还设计了一种针对PF解码器的水印方案，能够保持样本的分布不变，并实现任意低的假阳性率和高的召回率。实验证明PF解码器在困惑度方面明显优于朴素采样，为LLM解码提供了一种有希望的新方法。

    

    在本文中，我们提出了一种名为Permute-and-Flip（PF）解码器的新解码方法。它具有与标准采样解码器相似的鲁棒性特性，但在质量和鲁棒性的 tradeoff 上证明比采样方法更好，且永远不会差于任何其他解码器。同时，我们还设计了一种类似于Aaronson的Gumbel水印的加密水印方案，但是针对PF解码器而自然量身定制。该水印方案不改变样本的分布，同时允许任意低的假阳性率和高的召回率，只要生成的文本具有高熵。我们的实验证明，PF解码器（及其带有水印的对应物）在困惑度方面明显优于朴素采样（及其带有Gumbel水印的对应物），同时保持相同的鲁棒性（和可检测性），因此为LLM解码提供了一个有希望的新方法。代码可在https://github.com/XuandongZhao/pf-decoding找到。

    In this paper, we propose a new decoding method called Permute-and-Flip (PF) decoder. It enjoys robustness properties similar to the standard sampling decoder, but is provably up to 2x better in its quality-robustness tradeoff than sampling and never worse than any other decoder. We also design a cryptographic watermarking scheme analogous to Aaronson's Gumbel watermark, but naturally tailored for PF decoder. The watermarking scheme does not change the distribution to sample, while allowing arbitrarily low false positive rate and high recall whenever the generated text has high entropy. Our experiments show that the PF decoder (and its watermarked counterpart) significantly outperform(s) naive sampling (and it's Gumbel watermarked counterpart) in terms of perplexity, while retaining the same robustness (and detectability), hence making it a promising new approach for LLM decoding. The code is available at https://github.com/XuandongZhao/pf-decoding
    
[^2]: 针对多出口网络的模型窃取攻击

    Model Stealing Attack against Multi-Exit Networks. (arXiv:2305.13584v1 [cs.CR])

    [http://arxiv.org/abs/2305.13584](http://arxiv.org/abs/2305.13584)

    该论文介绍了第一个能同时窃取多出口网络模型函数和输出策略的攻击方法，并使用贝叶斯变点检测和性能损失、策略损失指导替代模型的训练。开发了一种新的输出策略搜索方法。

    

    与具有单个出口的传统神经网络相比，多出口网络具有多个出口，这些出口允许从模型的中间层早期输出，从而在保持类似识别精度的情况下提高计算效率。当使用传统的模型窃取攻击方法尝试窃取这些有价值的模型时，我们发现传统方法只能窃取模型的分类函数，而不能捕捉其输出策略。这导致窃取的替代模型的计算效率显著降低，失去多出口网络的优点。在本文中，我们提出了第一个窃取模型攻击，可以提取模型函数和输出策略。我们采用贝叶斯变点检测来分析目标模型的输出策略，并使用性能损失和策略损失来指导替代模型的训练。此外，我们设计了一种新颖的输出策略搜索方法，以使替代模型还原窃取目标模型的输出策略。

    Compared to traditional neural networks with a single exit, a multi-exit network has multiple exits that allow for early output from intermediate layers of the model, thus bringing significant improvement in computational efficiency while maintaining similar recognition accuracy. When attempting to steal such valuable models using traditional model stealing attacks, we found that conventional methods can only steal the model's classification function while failing to capture its output strategy. This results in a significant decrease in computational efficiency for the stolen substitute model, thereby losing the advantages of multi-exit networks.In this paper, we propose the first model stealing attack to extract both the model function and output strategy. We employ bayesian changepoint detection to analyze the target model's output strategy and use performance loss and strategy loss to guide the training of the substitute model. Furthermore, we designed a novel output strategy search
    

