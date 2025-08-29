# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FLASH: Federated Learning Across Simultaneous Heterogeneities](https://arxiv.org/abs/2402.08769) | FLASH是一个跨同时异质性的联邦学习方法，通过综合考虑数据质量、数据分布和延迟等因素，优于其他现有的方法。 |

# 详细

[^1]: FLASH: 跨同时异质性的联邦学习

    FLASH: Federated Learning Across Simultaneous Heterogeneities

    [https://arxiv.org/abs/2402.08769](https://arxiv.org/abs/2402.08769)

    FLASH是一个跨同时异质性的联邦学习方法，通过综合考虑数据质量、数据分布和延迟等因素，优于其他现有的方法。

    

    联邦学习（FL）的关键前提是在不交换本地数据的情况下，训练机器学习模型在不同的数据所有者（客户端）之间。迄今为止，这个方法面临的一个重要挑战是客户端的异质性，这可能不仅来自数据分布的变化，还来自数据质量以及计算/通信延迟方面的差异。对这些不同且同时存在的异质性的综合视图至关重要；例如，延迟较低的客户端可能具有较差的数据质量，反之亦然。在这项工作中，我们提出了FLASH（跨同时异质性的联邦学习），一个轻量且灵活的客户端选择算法，通过权衡与客户端数据质量、数据分布和延迟相关的统计信息，优于最先进的FL框架。据我们所知，FLASH是第一个能够在统一的方法中处理所有这些异质性的方法。

    arXiv:2402.08769v1 Announce Type: new Abstract: The key premise of federated learning (FL) is to train ML models across a diverse set of data-owners (clients), without exchanging local data. An overarching challenge to this date is client heterogeneity, which may arise not only from variations in data distribution, but also in data quality, as well as compute/communication latency. An integrated view of these diverse and concurrent sources of heterogeneity is critical; for instance, low-latency clients may have poor data quality, and vice versa. In this work, we propose FLASH(Federated Learning Across Simultaneous Heterogeneities), a lightweight and flexible client selection algorithm that outperforms state-of-the-art FL frameworks under extensive sources of heterogeneity, by trading-off the statistical information associated with the client's data quality, data distribution, and latency. FLASH is the first method, to our knowledge, for handling all these heterogeneities in a unified m
    

