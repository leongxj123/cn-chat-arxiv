# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Shrinkage Estimation of Fixed Effects in Linear Panel Data Models.](http://arxiv.org/abs/2308.12485) | 本文提出了一种在线性面板数据模型中估计固定效应的最优缩小估计方法，该方法不需要分布假设，并能够充分地利用序列相关性和时间变化。同时，还提供了一种预测未来固定效应的方法。 |
| [^2] | [Adaptive Student's t-distribution with method of moments moving estimator for nonstationary time series.](http://arxiv.org/abs/2304.03069) | 本文提出了一种适用于非平稳时间序列的自适应学生t分布方法，基于方法的一般自适应矩可以使用廉价的指数移动平均值（EMA）来估计参数。 |

# 详细

[^1]: 线性面板数据模型中固定效应最优缩小估计

    Optimal Shrinkage Estimation of Fixed Effects in Linear Panel Data Models. (arXiv:2308.12485v1 [econ.EM])

    [http://arxiv.org/abs/2308.12485](http://arxiv.org/abs/2308.12485)

    本文提出了一种在线性面板数据模型中估计固定效应的最优缩小估计方法，该方法不需要分布假设，并能够充分地利用序列相关性和时间变化。同时，还提供了一种预测未来固定效应的方法。

    

    缩小估计方法经常被用于估计固定效应，以减少最小二乘估计的噪声。然而，广泛使用的缩小估计仅在强分布假设下才能保证降低噪声。本文开发了一种估计固定效应的估计器，在缩小估计器类别中获得了最佳的均方误差。该类别包括传统的缩小估计器，且最优性不需要分布假设。该估计器具有直观的形式，并且易于实现。此外，固定效应允许随时间变化，并且可以具有序列相关性，而缩小方法在这种情况下可以最优地结合底层相关结构。在这样的背景下，还提供了一种预测未来一个时期固定效应的方法。

    Shrinkage methods are frequently used to estimate fixed effects to reduce the noisiness of the least square estimators. However, widely used shrinkage estimators guarantee such noise reduction only under strong distributional assumptions. I develop an estimator for the fixed effects that obtains the best possible mean squared error within a class of shrinkage estimators. This class includes conventional shrinkage estimators and the optimality does not require distributional assumptions. The estimator has an intuitive form and is easy to implement. Moreover, the fixed effects are allowed to vary with time and to be serially correlated, and the shrinkage optimally incorporates the underlying correlation structure in this case. In such a context, I also provide a method to forecast fixed effects one period ahead.
    
[^2]: 自适应学生t分布与方法矩移动估计器用于非平稳时间序列

    Adaptive Student's t-distribution with method of moments moving estimator for nonstationary time series. (arXiv:2304.03069v1 [stat.ME])

    [http://arxiv.org/abs/2304.03069](http://arxiv.org/abs/2304.03069)

    本文提出了一种适用于非平稳时间序列的自适应学生t分布方法，基于方法的一般自适应矩可以使用廉价的指数移动平均值（EMA）来估计参数。

    

    真实的时间序列通常是非平稳的，这带来了模型适应的难题。传统方法如GARCH假定任意类型的依赖性。为了避免这种偏差，我们将着眼于最近提出的不可知的移动估计器哲学：在时间$t$找到优化$F_t=\sum_{\tau<t} (1-\eta)^{t-\tau} \ln(\rho_\theta (x_\tau))$移动对数似然的参数，随时间演化。例如，它允许使用廉价的指数移动平均值（EMA）来估计参数，例如绝对中心矩$E[|x-\mu|^p]$随$p\in\mathbb{R}^+$的变化而演化$m_{p,t+1} = m_{p,t} + \eta (|x_t-\mu_t|^p-m_{p,t})$。这种基于方法的一般自适应矩的应用将呈现在学生t分布上，尤其是在经济应用中流行，这里应用于DJIA公司的对数收益率。

    The real life time series are usually nonstationary, bringing a difficult question of model adaptation. Classical approaches like GARCH assume arbitrary type of dependence. To prevent such bias, we will focus on recently proposed agnostic philosophy of moving estimator: in time $t$ finding parameters optimizing e.g. $F_t=\sum_{\tau<t} (1-\eta)^{t-\tau} \ln(\rho_\theta (x_\tau))$ moving log-likelihood, evolving in time. It allows for example to estimate parameters using inexpensive exponential moving averages (EMA), like absolute central moments $E[|x-\mu|^p]$ evolving with $m_{p,t+1} = m_{p,t} + \eta (|x_t-\mu_t|^p-m_{p,t})$ for one or multiple powers $p\in\mathbb{R}^+$. Application of such general adaptive methods of moments will be presented on Student's t-distribution, popular especially in economical applications, here applied to log-returns of DJIA companies.
    

