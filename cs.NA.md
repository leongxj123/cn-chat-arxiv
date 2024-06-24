# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Predictions Based on Pixel Data: Insights from PDEs and Finite Differences.](http://arxiv.org/abs/2305.00723) | 本文介绍了基于像素数据的预测，通过对离散卷积和有限差分算子之间联系的利用，证明了逼近自偏微分方程空时离散出的序列可以使用相对较小的卷积(残差)网络进行。 |

# 详细

[^1]: 基于像素数据的预测: PDE和有限差分的深入探索

    Predictions Based on Pixel Data: Insights from PDEs and Finite Differences. (arXiv:2305.00723v1 [math.NA])

    [http://arxiv.org/abs/2305.00723](http://arxiv.org/abs/2305.00723)

    本文介绍了基于像素数据的预测，通过对离散卷积和有限差分算子之间联系的利用，证明了逼近自偏微分方程空时离散出的序列可以使用相对较小的卷积(残差)网络进行。

    

    神经网络是高维空间中许多逼近任务的最先进技术，这得到了大量实验证据的支持。然而，我们仍需要对它们可以逼近的内容以及以何种代价和精度逼近有一个坚实的理论理解。其中一个在涉及图像的逼近任务中有实际用途的网络体系结构是卷积(残差)网络。然而，由于这些网络中涉及的线性算子的局部性质，它们的分析比通用全连接神经网络更为复杂。本文重点介绍的是序列逼近任务，其中每个观察值由矩阵或高阶张量表示。我们证明，当逼近自偏微分方程空时离散出的序列时，可以使用相对较小的网络。我们通过利用离散卷积和有限差分算子之间的联系来构造这些结果。在整个过程中，我们设计了我们的网络。

    Neural networks are the state-of-the-art for many approximation tasks in high-dimensional spaces, as supported by an abundance of experimental evidence. However, we still need a solid theoretical understanding of what they can approximate and, more importantly, at what cost and accuracy. One network architecture of practical use, especially for approximation tasks involving images, is convolutional (residual) networks. However, due to the locality of the linear operators involved in these networks, their analysis is more complicated than for generic fully connected neural networks. This paper focuses on sequence approximation tasks, where a matrix or a higher-order tensor represents each observation. We show that when approximating sequences arising from space-time discretisations of PDEs we may use relatively small networks. We constructively derive these results by exploiting connections between discrete convolution and finite difference operators. Throughout, we design our network a
    

