# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Land-cover change detection using paired OpenStreetMap data and optical high-resolution imagery via object-guided Transformer.](http://arxiv.org/abs/2310.02674) | 本文通过直接利用配对的OSM数据和光学图像进行土地覆盖变化检测，提出了一种基于对象引导的Transformer架构，从而拓宽了变化检测任务的范围，并显著减少了计算开销和内存负担。 |

# 详细

[^1]: 利用配对的OpenStreetMap数据和光学高分辨率影像进行土地覆盖变化检测

    Land-cover change detection using paired OpenStreetMap data and optical high-resolution imagery via object-guided Transformer. (arXiv:2310.02674v1 [cs.CV])

    [http://arxiv.org/abs/2310.02674](http://arxiv.org/abs/2310.02674)

    本文通过直接利用配对的OSM数据和光学图像进行土地覆盖变化检测，提出了一种基于对象引导的Transformer架构，从而拓宽了变化检测任务的范围，并显著减少了计算开销和内存负担。

    

    光学高分辨率影像和OpenStreetMap（OSM）数据是土地覆盖变化检测的两个重要数据源。先前的研究主要利用OSM数据来辅助多时期光学高分辨率图像的变化检测。本文通过直接利用配对的OSM数据和光学图像进行土地覆盖变化检测，拓宽了变化检测任务的范围，涵盖更多动态地球观测。为此，我们提出了一种基于对象引导的Transformer（ObjFormer）架构，将流行的基于对象的图像分析（OBIA）技术与先进的视觉Transformer架构自然地结合起来。引入OBIA可以显著减少自注意力模块中的计算开销和内存负担。具体而言，所提出的ObjFormer具有层次伪孪生编码器，包含对象引导自注意力模块，用于提取代表性特征。

    Optical high-resolution imagery and OpenStreetMap (OSM) data are two important data sources for land-cover change detection. Previous studies in these two data sources focus on utilizing the information in OSM data to aid the change detection on multi-temporal optical high-resolution images. This paper pioneers the direct detection of land-cover changes utilizing paired OSM data and optical imagery, thereby broadening the horizons of change detection tasks to encompass more dynamic earth observations. To this end, we propose an object-guided Transformer (ObjFormer) architecture by naturally combining the prevalent object-based image analysis (OBIA) technique with the advanced vision Transformer architecture. The introduction of OBIA can significantly reduce the computational overhead and memory burden in the self-attention module. Specifically, the proposed ObjFormer has a hierarchical pseudo-siamese encoder consisting of object-guided self-attention modules that extract representative
    

