# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Near-realtime Facial Animation by Deep 3D Simulation Super-Resolution.](http://arxiv.org/abs/2305.03216) | 该论文提出了一种基于神经网络的3D模拟超分辨率框架，能够高效、逼真地增强低成本、实时物理模拟产生的面部表现，使其接近于具有更高分辨率和准确物理建模的参考质量离线模拟器。 |

# 详细

[^1]: 基于神经网络的3D模拟超分辨率实现近实时面部动画表现

    Near-realtime Facial Animation by Deep 3D Simulation Super-Resolution. (arXiv:2305.03216v1 [cs.GR])

    [http://arxiv.org/abs/2305.03216](http://arxiv.org/abs/2305.03216)

    该论文提出了一种基于神经网络的3D模拟超分辨率框架，能够高效、逼真地增强低成本、实时物理模拟产生的面部表现，使其接近于具有更高分辨率和准确物理建模的参考质量离线模拟器。

    

    我们提出了一种基于神经网络的模拟超分辨率框架，能够高效、逼真地增强低成本、实时物理模拟产生的面部表现，使其接近于具有更高分辨率（在我们的实验中高达26倍的元素数）和准确物理建模的参考质量离线模拟器。我们的方法源于我们通过模拟构建一组配对帧序列的能力，这些序列分别来自于低分辨率和高分辨率模拟器，并且在语义上相互对应。我们以面部动画为例，创造这种语义一致性的方式就是在两个模拟器中调整同样的肌肉激活控制和骨架姿势。我们提出的神经网络超分辨率框架从这个训练集中泛化到看不见的表情，并且补偿两个模拟之间的建模差异。

    We present a neural network-based simulation super-resolution framework that can efficiently and realistically enhance a facial performance produced by a low-cost, realtime physics-based simulation to a level of detail that closely approximates that of a reference-quality off-line simulator with much higher resolution (26x element count in our examples) and accurate physical modeling. Our approach is rooted in our ability to construct - via simulation - a training set of paired frames, from the low- and high-resolution simulators respectively, that are in semantic correspondence with each other. We use face animation as an exemplar of such a simulation domain, where creating this semantic congruence is achieved by simply dialing in the same muscle actuation controls and skeletal pose in the two simulators. Our proposed neural network super-resolution framework generalizes from this training set to unseen expressions, compensates for modeling discrepancies between the two simulations du
    

