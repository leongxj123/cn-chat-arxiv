# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting](https://arxiv.org/abs/2403.08551) | 通过2D高斯喷涂实现图像表示和压缩，在GPU内存占用降低的情况下，提供了更快的渲染速度，并在表示性能上与INR相匹敌。 |

# 详细

[^1]: 高斯图像：通过2D高斯喷涂进行1000帧每秒的图像表示和压缩

    GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting

    [https://arxiv.org/abs/2403.08551](https://arxiv.org/abs/2403.08551)

    通过2D高斯喷涂实现图像表示和压缩，在GPU内存占用降低的情况下，提供了更快的渲染速度，并在表示性能上与INR相匹敌。

    

    最近，隐式神经表示（INR）在图像表示和压缩方面取得了巨大成功，提供了高视觉质量和快速渲染速度，每秒10-1000帧，假设有足够的GPU资源可用。然而，这种要求常常阻碍了它们在内存有限的低端设备上的使用。为此，我们提出了一种通过2D高斯喷涂进行图像表示和压缩的开创性范式，名为GaussianImage。我们首先引入2D高斯来表示图像，其中每个高斯具有8个参数，包括位置、协方差和颜色。随后，我们揭示了一种基于累积求和的新颖渲染算法。值得注意的是，我们的方法使用GPU内存至少降低3倍，拟合时间快5倍，不仅在表示性能上与INR（例如WIRE，I-NGP）不相上下，而且无论参数大小如何都能提供1500-2000帧每秒的更快渲染速度。

    arXiv:2403.08551v1 Announce Type: cross  Abstract: Implicit neural representations (INRs) recently achieved great success in image representation and compression, offering high visual quality and fast rendering speeds with 10-1000 FPS, assuming sufficient GPU resources are available. However, this requirement often hinders their use on low-end devices with limited memory. In response, we propose a groundbreaking paradigm of image representation and compression by 2D Gaussian Splatting, named GaussianImage. We first introduce 2D Gaussian to represent the image, where each Gaussian has 8 parameters including position, covariance and color. Subsequently, we unveil a novel rendering algorithm based on accumulated summation. Remarkably, our method with a minimum of 3$\times$ lower GPU memory usage and 5$\times$ faster fitting time not only rivals INRs (e.g., WIRE, I-NGP) in representation performance, but also delivers a faster rendering speed of 1500-2000 FPS regardless of parameter size. 
    

