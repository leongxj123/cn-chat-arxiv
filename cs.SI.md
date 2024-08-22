# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PathMLP: Smooth Path Towards High-order Homophily.](http://arxiv.org/abs/2306.13532) | 本文提出了 PathMLP，一种基于相似性路径采样策略的轻量化模型，它能够克服传统 GNNs 中在获得高阶信息过程中存在的缺陷，即过度平滑问题、高阶信息未充分利用以及计算效率低等问题，并能够利用高阶信息进行节点表示学习。 |

# 详细

[^1]: PathMLP：高阶同质性的平滑路径

    PathMLP: Smooth Path Towards High-order Homophily. (arXiv:2306.13532v1 [cs.LG])

    [http://arxiv.org/abs/2306.13532](http://arxiv.org/abs/2306.13532)

    本文提出了 PathMLP，一种基于相似性路径采样策略的轻量化模型，它能够克服传统 GNNs 中在获得高阶信息过程中存在的缺陷，即过度平滑问题、高阶信息未充分利用以及计算效率低等问题，并能够利用高阶信息进行节点表示学习。

    

    实际的图表现出越来越多的异质性，节点不再倾向于连接具有相同标签的节点，挑战了经典图神经网络(GNNs)的同质性假设并阻碍了它们的性能。有趣的是，我们观察到某些异质数据的高阶信息表现出高同质性，这促使我们在节点表示学习中涉及高阶信息。然而，GNNs中获得高阶信息的常见做法主要通过增加模型深度和改变消息传递机制，虽然在一定程度上是有效的，但它们存在三个缺点：1）由于过度的模型深度和传播时间而过度平滑; 2）高阶信息没有充分利用; 3）计算效率低。因此，我们设计了一种基于相似性的路径采样策略，用于捕获包含高阶同质性的平滑路径。然后，我们提出了一个基于多层感知器(Multi-layer Perceptrons, MLP)的轻量化模型，称之为PathMLP。

    Real-world graphs exhibit increasing heterophily, where nodes no longer tend to be connected to nodes with the same label, challenging the homophily assumption of classical graph neural networks (GNNs) and impeding their performance. Intriguingly, we observe that certain high-order information on heterophilous data exhibits high homophily, which motivates us to involve high-order information in node representation learning. However, common practices in GNNs to acquire high-order information mainly through increasing model depth and altering message-passing mechanisms, which, albeit effective to a certain extent, suffer from three shortcomings: 1) over-smoothing due to excessive model depth and propagation times; 2) high-order information is not fully utilized; 3) low computational efficiency. In this regard, we design a similarity-based path sampling strategy to capture smooth paths containing high-order homophily. Then we propose a lightweight model based on multi-layer perceptrons (M
    

