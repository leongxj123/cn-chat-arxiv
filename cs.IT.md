# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal vintage factor analysis with deflation varimax.](http://arxiv.org/abs/2310.10545) | 本文提出了一种采用通货紧缩变量旋转的拟合因子分析方法，在每一行上逐步求解正交矩阵，相比于传统方法具有更好的计算性能和灵活性，并且在更广泛的背景下提供了理论保证。 |

# 详细

[^1]: 优化拟合因子分析与通货紧缩变量旋转

    Optimal vintage factor analysis with deflation varimax. (arXiv:2310.10545v1 [stat.ML])

    [http://arxiv.org/abs/2310.10545](http://arxiv.org/abs/2310.10545)

    本文提出了一种采用通货紧缩变量旋转的拟合因子分析方法，在每一行上逐步求解正交矩阵，相比于传统方法具有更好的计算性能和灵活性，并且在更广泛的背景下提供了理论保证。

    

    通货紧缩变量旋转是一种重要的因子分析方法，旨在首先找到原始数据的低维表示，然后寻求旋转，使旋转后的低维表示具有科学意义。尽管Principal Component Analysis (PCA) followed by the varimax rotation被广泛应用于拟合因子分析，但由于varimax rotation需要在正交矩阵集合上解非凸优化问题，因此很难提供理论保证。本文提出了一种逐行求解正交矩阵的通货紧缩变量旋转过程。除了在计算上的优势和灵活性之外，我们还能在广泛的背景下对所提出的过程进行完全的理论保证。在PCA之后采用这种新的varimax方法作为第二步，我们进一步分析了这个两步过程在一个更一般的因子模型的情况下。

    Vintage factor analysis is one important type of factor analysis that aims to first find a low-dimensional representation of the original data, and then to seek a rotation such that the rotated low-dimensional representation is scientifically meaningful. Perhaps the most widely used vintage factor analysis is the Principal Component Analysis (PCA) followed by the varimax rotation. Despite its popularity, little theoretical guarantee can be provided mainly because varimax rotation requires to solve a non-convex optimization over the set of orthogonal matrices.  In this paper, we propose a deflation varimax procedure that solves each row of an orthogonal matrix sequentially. In addition to its net computational gain and flexibility, we are able to fully establish theoretical guarantees for the proposed procedure in a broad context.  Adopting this new varimax approach as the second step after PCA, we further analyze this two step procedure under a general class of factor models. Our resul
    

