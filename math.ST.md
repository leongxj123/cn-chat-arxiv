# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data-Driven Tuning Parameter Selection for High-Dimensional Vector Autoregressions](https://arxiv.org/abs/2403.06657) | 通过提出一种基于数据驱动的加权Lasso估计器，解决高维向量自回归模型中调参选择问题。 |

# 详细

[^1]: 基于数据驱动的高维向量自回归调参选择

    Data-Driven Tuning Parameter Selection for High-Dimensional Vector Autoregressions

    [https://arxiv.org/abs/2403.06657](https://arxiv.org/abs/2403.06657)

    通过提出一种基于数据驱动的加权Lasso估计器，解决高维向量自回归模型中调参选择问题。

    

    Lasso类型的估计器通常用于估计高维时间序列模型。Lasso的理论保证通常要求选择合适的惩罚水平，这通常取决于未知的总体数量。然而，得到的估计值和模型中保留的变量数量关键取决于所选择的惩罚水平。目前，在高维时间序列的情况下没有理论上的指导来做这个选择。我们通过考虑也许是最常用的多元时间序列模型之一，线性向量自回归（VAR）模型的估计来解决这个问题，并提出了一个以完全数据驱动方式选择惩罚的加权Lasso估计量。我们为这一方法建立的理论保证

    arXiv:2403.06657v1 Announce Type: new  Abstract: Lasso-type estimators are routinely used to estimate high-dimensional time series models. The theoretical guarantees established for Lasso typically require the penalty level to be chosen in a suitable fashion often depending on unknown population quantities. Furthermore, the resulting estimates and the number of variables retained in the model depend crucially on the chosen penalty level. However, there is currently no theoretically founded guidance for this choice in the context of high-dimensional time series. Instead one resorts to selecting the penalty level in an ad hoc manner using, e.g., information criteria or cross-validation. We resolve this problem by considering estimation of the perhaps most commonly employed multivariate time series model, the linear vector autoregressive (VAR) model, and propose a weighted Lasso estimator with penalization chosen in a fully data-driven way. The theoretical guarantees that we establish for
    

