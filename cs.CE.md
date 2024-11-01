# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Linking Across Data Granularity: Fitting Multivariate Hawkes Processes to Partially Interval-Censored Data.](http://arxiv.org/abs/2111.02062) | 这项研究介绍了Partial Mean Behavior Poisson (PMBP)过程，它是一种新的点过程，可以有效地建模时间戳和区间屏蔽数据，并成功恢复了MHP参数和谱半径。 |

# 详细

[^1]: 跨数据粒度链接：拟合多变量Hawkes过程到部分区间屏蔽数据

    Linking Across Data Granularity: Fitting Multivariate Hawkes Processes to Partially Interval-Censored Data. (arXiv:2111.02062v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2111.02062](http://arxiv.org/abs/2111.02062)

    这项研究介绍了Partial Mean Behavior Poisson (PMBP)过程，它是一种新的点过程，可以有效地建模时间戳和区间屏蔽数据，并成功恢复了MHP参数和谱半径。

    

    多变量Hawkes过程(MHP)被广泛用于分析相互作用的数据流，其中事件在自身维度内（通过自激）或不同维度之间（通过交叉激发）生成新事件。然而，在某些应用中，某些维度中的个别事件时间戳是不可观测的，只有在时间间隔内的事件计数是已知的，被称为部分区间屏蔽数据。MHP不适用于处理这种数据，因为其估计需要事件时间戳。在本研究中，我们引入了Partial Mean Behavior Poisson (PMBP)过程，这是一种新颖的点过程，与MHP共享参数等效性，可以有效地建模时间戳和区间屏蔽数据。我们使用合成和真实世界数据集展示了PMBP过程的能力。首先，我们证明了PMBP过程可以近似MHP参数并恢复谱半径，使用合成事件历史数据进行验证。

    The multivariate Hawkes process (MHP) is widely used for analyzing data streams that interact with each other, where events generate new events within their own dimension (via self-excitation) or across different dimensions (via cross-excitation). However, in certain applications, the timestamps of individual events in some dimensions are unobservable, and only event counts within intervals are known, referred to as partially interval-censored data. The MHP is unsuitable for handling such data since its estimation requires event timestamps. In this study, we introduce the Partial Mean Behavior Poisson (PMBP) process, a novel point process which shares parameter equivalence with the MHP and can effectively model both timestamped and interval-censored data. We demonstrate the capabilities of the PMBP process using synthetic and real-world datasets. Firstly, we illustrate that the PMBP process can approximate MHP parameters and recover the spectral radius using synthetic event histories. 
    

