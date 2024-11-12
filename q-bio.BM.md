# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enabling Efficient Equivariant Operations in the Fourier Basis via Gaunt Tensor Products.](http://arxiv.org/abs/2401.10216) | 该论文提出了一种加速计算不可约表示张量积的方法，通过将等变操作基础从球形谐波改变为2D傅立叶基础，实现了对E(3)群的等变神经网络的高效建模。 |

# 详细

[^1]: 通过Gaunt张量积在傅里叶基础上实现高效的等变操作

    Enabling Efficient Equivariant Operations in the Fourier Basis via Gaunt Tensor Products. (arXiv:2401.10216v1 [cs.LG])

    [http://arxiv.org/abs/2401.10216](http://arxiv.org/abs/2401.10216)

    该论文提出了一种加速计算不可约表示张量积的方法，通过将等变操作基础从球形谐波改变为2D傅立叶基础，实现了对E(3)群的等变神经网络的高效建模。

    

    在建模现实世界应用中的3D数据时，发展E(3)群的等变神经网络起着重要作用。实现这种等变性主要涉及到不可约表示（irreps）的张量积。然而，随着使用高阶张量，这些操作的计算复杂性显著增加。在这项工作中，我们提出了一种系统的方法来大大加速不可约表示的张量积的计算。我们将常用的Clebsch-Gordan系数与Gaunt系数进行了数学上的连接，Gaunt系数是三个球形谐波乘积的积分。通过Gaunt系数，不可约表示的张量积等价于由球形谐波表示的球形函数之间的乘法。这种观点进一步使我们能够将等变操作的基础从球形谐波改变为2D傅立叶基础。因此，球形函数之间的乘法可以在傅立叶基础上进行。

    Developing equivariant neural networks for the E(3) group plays an important role in modeling 3D data across real-world applications. Enforcing this equivariance primarily involves the tensor products of irreducible representations (irreps). However, the computational complexity of such operations increases significantly as higher-order tensors are used. In this work, we propose a systematic approach to substantially accelerate the computation of the tensor products of irreps. We mathematically connect the commonly used Clebsch-Gordan coefficients to the Gaunt coefficients, which are integrals of products of three spherical harmonics. Through Gaunt coefficients, the tensor product of irreps becomes equivalent to the multiplication between spherical functions represented by spherical harmonics. This perspective further allows us to change the basis for the equivariant operations from spherical harmonics to a 2D Fourier basis. Consequently, the multiplication between spherical functions 
    

