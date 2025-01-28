# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DiFaReli : Diffusion Face Relighting.](http://arxiv.org/abs/2304.09479) | DiFaReli提出了一种新方法，通过利用条件扩散隐式模型解码解耦的光编码以及从现成的估算器推断出的与3D形状和面部身份相关的其他编码，能够处理单视角的野外环境下的人脸重照，无需光线舞台数据、多视图图像或光照基础事实，实验表明其效果优于现有方法。 |

# 详细

[^1]: DiFaReli: 扩散人脸重照技术

    DiFaReli : Diffusion Face Relighting. (arXiv:2304.09479v1 [cs.CV])

    [http://arxiv.org/abs/2304.09479](http://arxiv.org/abs/2304.09479)

    DiFaReli提出了一种新方法，通过利用条件扩散隐式模型解码解耦的光编码以及从现成的估算器推断出的与3D形状和面部身份相关的其他编码，能够处理单视角的野外环境下的人脸重照，无需光线舞台数据、多视图图像或光照基础事实，实验表明其效果优于现有方法。

    

    我们提出了一种新方法，用于处理野外环境下的单视角人脸重照。处理全局照明或投影阴影等非漫反射效应一直是人脸重照领域的难点。以往的研究通常假定兰伯特反射表面，简化光照模型，或者需要估计三维形状、反射率或阴影图。然而，这种估计是容易出错的，需要许多具有光照基础事实的训练样本才能很好地推广。我们的研究绕过了准确估计固有组件的需要，可以仅通过2D图像训练而不需要任何光线舞台数据、多视图图像或光照基础事实。我们的关键思想是利用条件扩散隐式模型（DDIM）解码解耦的光编码以及从现成的估算器推断出的与3D形状和面部身份相关的其他编码。我们还提出了一种新的调节技术，通过使用归一化方案，简化光与几何之间复杂互动的建模。在多个基准数据集上的实验表明，我们的方法优于现有方法。

    We present a novel approach to single-view face relighting in the wild. Handling non-diffuse effects, such as global illumination or cast shadows, has long been a challenge in face relighting. Prior work often assumes Lambertian surfaces, simplified lighting models or involves estimating 3D shape, albedo, or a shadow map. This estimation, however, is error-prone and requires many training examples with lighting ground truth to generalize well. Our work bypasses the need for accurate estimation of intrinsic components and can be trained solely on 2D images without any light stage data, multi-view images, or lighting ground truth. Our key idea is to leverage a conditional diffusion implicit model (DDIM) for decoding a disentangled light encoding along with other encodings related to 3D shape and facial identity inferred from off-the-shelf estimators. We also propose a novel conditioning technique that eases the modeling of the complex interaction between light and geometry by using a ren
    

