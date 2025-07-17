# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Understanding Pan-Sharpening via Generalized Inverse.](http://arxiv.org/abs/2310.02718) | 通过研究广义逆理论，本文提出了一种新的全色增强算法，该算法基于简单矩阵方程描述全色增强问题，并探讨解的条件和光谱、空间分辨率的获取。通过引入降采样增强方法，我们得到了与分量替代和多分辨率分析方法相对应的广义逆矩阵表达式，并提出了一个新的模型先验来解决全色增强中的理论误差问题。 |

# 详细

[^1]: 通过广义逆理解全色增强算法

    Understanding Pan-Sharpening via Generalized Inverse. (arXiv:2310.02718v1 [cs.LG])

    [http://arxiv.org/abs/2310.02718](http://arxiv.org/abs/2310.02718)

    通过研究广义逆理论，本文提出了一种新的全色增强算法，该算法基于简单矩阵方程描述全色增强问题，并探讨解的条件和光谱、空间分辨率的获取。通过引入降采样增强方法，我们得到了与分量替代和多分辨率分析方法相对应的广义逆矩阵表达式，并提出了一个新的模型先验来解决全色增强中的理论误差问题。

    

    全色增强算法利用全色图像和多光谱图像获取具有高空间和高光谱的图像。然而，这些算法的优化是根据不同的标准设计的。我们采用简单的矩阵方程来描述全色增强问题，并讨论解的存在条件以及光谱和空间分辨率的获取。我们引入了一种降采样增强方法，以更好地获取空间和光谱降采样矩阵。通过广义逆理论，我们推导出了两种形式的广义逆矩阵表达式，可以对应于两个主要的全色增强方法：分量替代和多分辨率分析方法。具体而言，我们证明了Gram Schmidt自适应(GSA)方法遵循分量替代的广义逆矩阵表达式。我们提出了一个在光谱函数的广义逆矩阵之前的模型先验。我们对理论误差进行了分析。

    Pan-sharpening algorithm utilizes panchromatic image and multispectral image to obtain a high spatial and high spectral image. However, the optimizations of the algorithms are designed with different standards. We adopt the simple matrix equation to describe the Pan-sharpening problem. The solution existence condition and the acquirement of spectral and spatial resolution are discussed. A down-sampling enhancement method was introduced for better acquiring the spatial and spectral down-sample matrices. By the generalized inverse theory, we derived two forms of general inverse matrix formulations that can correspond to the two prominent classes of Pan-sharpening methods, that is, component substitution and multi-resolution analysis methods. Specifically, the Gram Schmidt Adaptive(GSA) was proved to follow the general inverse matrix formulation of component substitution. A model prior to the general inverse matrix of the spectral function was rendered. The theoretical errors are analyzed
    

