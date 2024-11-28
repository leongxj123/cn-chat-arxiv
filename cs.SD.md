# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Say Goodbye to RNN-T Loss: A Novel CIF-based Transducer Architecture for Automatic Speech Recognition.](http://arxiv.org/abs/2307.14132) | 本文提出了一种名为CIF-Transducer的新型模型，将连续积分和火机制与RNN-T模型结合起来，实现了高效的对齐，并放弃了RNN-T Loss，从而减少了计算量，并使预测网络发挥更重要的作用。实验证明CIF-T在自动语音识别中取得了最先进的结果。 |

# 详细

[^1]: 告别RNN-T Loss：一种新的基于CIF的转录器架构用于自动语音识别

    Say Goodbye to RNN-T Loss: A Novel CIF-based Transducer Architecture for Automatic Speech Recognition. (arXiv:2307.14132v1 [cs.SD])

    [http://arxiv.org/abs/2307.14132](http://arxiv.org/abs/2307.14132)

    本文提出了一种名为CIF-Transducer的新型模型，将连续积分和火机制与RNN-T模型结合起来，实现了高效的对齐，并放弃了RNN-T Loss，从而减少了计算量，并使预测网络发挥更重要的作用。实验证明CIF-T在自动语音识别中取得了最先进的结果。

    

    RNN-T模型在ASR中广泛使用，依靠RNN-T Loss实现输入音频和目标序列的长度对齐。然而，RNN-T Loss的实现复杂性和基于对齐的优化目标导致计算冗余和预测网络角色的减少。在本文中，我们提出了一种名为CIF-Transducer（CIF-T）的新型模型，它将连续积分和火（CIF）机制与RNN-T模型结合起来，实现高效的对齐。通过这种方式，放弃了RNN-T Loss，从而减少了计算量，并使预测网络发挥更重要的作用。我们还引入了Funnel-CIF、Context Blocks、Unified Gating和Bilinear Pooling联合网络以及辅助训练策略来进一步提高性能。在178小时的AISHELL-1和10000小时的WenetSpeech数据集上的实验证明，与RNN-T模型相比，CIF-T以更低的计算开销实现了最先进的结果。

    RNN-T models are widely used in ASR, which rely on the RNN-T loss to achieve length alignment between input audio and target sequence. However, the implementation complexity and the alignment-based optimization target of RNN-T loss lead to computational redundancy and a reduced role for predictor network, respectively. In this paper, we propose a novel model named CIF-Transducer (CIF-T) which incorporates the Continuous Integrate-and-Fire (CIF) mechanism with the RNN-T model to achieve efficient alignment. In this way, the RNN-T loss is abandoned, thus bringing a computational reduction and allowing the predictor network a more significant role. We also introduce Funnel-CIF, Context Blocks, Unified Gating and Bilinear Pooling joint network, and auxiliary training strategy to further improve performance. Experiments on the 178-hour AISHELL-1 and 10000-hour WenetSpeech datasets show that CIF-T achieves state-of-the-art results with lower computational overhead compared to RNN-T models.
    

