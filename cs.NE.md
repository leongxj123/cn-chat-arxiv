# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Spiking NeRF: Making Bio-inspired Neural Networks See through the Real World.](http://arxiv.org/abs/2309.10987) | 本文介绍了SpikingNeRF，它通过将辐射光线与脉冲神经网络的时间维度对齐，以节省能量并减少计算量。 |

# 详细

[^1]: Spiking NeRF：使生物启发的神经网络穿透现实世界

    Spiking NeRF: Making Bio-inspired Neural Networks See through the Real World. (arXiv:2309.10987v1 [cs.NE])

    [http://arxiv.org/abs/2309.10987](http://arxiv.org/abs/2309.10987)

    本文介绍了SpikingNeRF，它通过将辐射光线与脉冲神经网络的时间维度对齐，以节省能量并减少计算量。

    

    脉冲神经网络（SNN）在许多任务中取得了成功，利用其具有潜在生物学可行性的能量效率和潜力。与此同时，神经辐射场（NeRF）以大量能量消耗渲染高质量的3D场景，但很少有研究深入探索以生物启发的方法进行节能解决方案。本文提出了脉冲NeRF（SpikingNeRF），将辐射光线与SNN的时间维度对齐，以自然地适应SNN对辐射场的重建。因此，计算以基于脉冲、无乘法的方式进行，从而减少能量消耗。在SpikingNeRF中，光线上的每个采样点匹配到特定的时间步，并以混合方式表示，其中体素网格也得到维护。基于体素网格，确定采样点是否在训练和推断过程中被屏蔽以进行更好的处理。然而，这个操作也会产生不可逆性。

    Spiking neuron networks (SNNs) have been thriving on numerous tasks to leverage their promising energy efficiency and exploit their potentialities as biologically plausible intelligence. Meanwhile, the Neural Radiance Fields (NeRF) render high-quality 3D scenes with massive energy consumption, and few works delve into the energy-saving solution with a bio-inspired approach. In this paper, we propose spiking NeRF (SpikingNeRF), which aligns the radiance ray with the temporal dimension of SNN, to naturally accommodate the SNN to the reconstruction of Radiance Fields. Thus, the computation turns into a spike-based, multiplication-free manner, reducing the energy consumption. In SpikingNeRF, each sampled point on the ray is matched onto a particular time step, and represented in a hybrid manner where the voxel grids are maintained as well. Based on the voxel grids, sampled points are determined whether to be masked for better training and inference. However, this operation also incurs irre
    

