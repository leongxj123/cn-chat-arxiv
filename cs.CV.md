# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Recent Advances in Multi-modal 3D Scene Understanding: A Comprehensive Survey and Evaluation.](http://arxiv.org/abs/2310.15676) | 多模态3D场景理解在自动驾驶和人机交互等领域中应用广泛。本文从理论的角度对多模态3D方法的进展进行了全面调查和评估。 |
| [^2] | [DeepIPCv2: LiDAR-powered Robust Environmental Perception and Navigational Control for Autonomous Vehicle.](http://arxiv.org/abs/2307.06647) | DeepIPCv2是一种利用LiDAR传感器感知环境的自动驾驶模型，通过使用点云作为感知输入，在各种条件下实现了更强大的驾驶性能。 |

# 详细

[^1]: 多模态3D场景理解的最新进展：一项全面调查和评估

    Recent Advances in Multi-modal 3D Scene Understanding: A Comprehensive Survey and Evaluation. (arXiv:2310.15676v1 [cs.CV])

    [http://arxiv.org/abs/2310.15676](http://arxiv.org/abs/2310.15676)

    多模态3D场景理解在自动驾驶和人机交互等领域中应用广泛。本文从理论的角度对多模态3D方法的进展进行了全面调查和评估。

    

    多模态3D场景理解因其在自动驾驶和人机交互等许多领域中的广泛应用而受到了广泛关注。与传统的单模态3D理解相比，引入额外的模态不仅提升了场景解释的丰富性和准确性，还确保了更稳健和弹性的理解。这在多样化和具有挑战性的环境中尤为关键，仅依靠3D数据可能是不足够的。尽管在过去三年中出现了许多多模态3D方法的发展，特别是那些整合多摄像头图像（3D+2D）和文本描述（3D+语言）的方法，但缺乏全面深入的评估。在本文中，我们对最近的进展进行了系统的调查，以填补这一空白。我们首先简要介绍了一个背景，正式定义了各种3D多模态任务，并总结了它们固有的挑战。

    Multi-modal 3D scene understanding has gained considerable attention due to its wide applications in many areas, such as autonomous driving and human-computer interaction. Compared to conventional single-modal 3D understanding, introducing an additional modality not only elevates the richness and precision of scene interpretation but also ensures a more robust and resilient understanding. This becomes especially crucial in varied and challenging environments where solely relying on 3D data might be inadequate. While there has been a surge in the development of multi-modal 3D methods over past three years, especially those integrating multi-camera images (3D+2D) and textual descriptions (3D+language), a comprehensive and in-depth review is notably absent. In this article, we present a systematic survey of recent progress to bridge this gap. We begin by briefly introducing a background that formally defines various 3D multi-modal tasks and summarizes their inherent challenges. After that
    
[^2]: DeepIPCv2：利用LiDAR强化自动驾驶环境感知与导航控制

    DeepIPCv2: LiDAR-powered Robust Environmental Perception and Navigational Control for Autonomous Vehicle. (arXiv:2307.06647v1 [cs.RO])

    [http://arxiv.org/abs/2307.06647](http://arxiv.org/abs/2307.06647)

    DeepIPCv2是一种利用LiDAR传感器感知环境的自动驾驶模型，通过使用点云作为感知输入，在各种条件下实现了更强大的驾驶性能。

    

    我们提出了DeepIPCv2，一种利用LiDAR传感器感知环境的自动驾驶模型，以实现更强大的驾驶性能，特别是在光照条件较差的情况下。DeepIPCv2使用一组LiDAR点云作为其主要感知输入。由于点云不受光照变化的影响，它们可以提供清晰的环境观察，无论条件如何。这使得感知模块能够提供更好的场景理解和稳定的特征，从而支持控制模块准确估计导航控制。为了评估其性能，我们通过部署该模型来预测一组驾驶记录并在三种不同条件下进行真实自动驾驶的测试。我们还进行了消融和比较研究，以证明其性能。基于实验结果，DeepIPCv2在所有条件下均显示出强大的驾驶性能。

    We present DeepIPCv2, an autonomous driving model that perceives the environment using a LiDAR sensor for more robust drivability, especially when driving under poor illumination conditions. DeepIPCv2 takes a set of LiDAR point clouds for its main perception input. As point clouds are not affected by illumination changes, they can provide a clear observation of the surroundings no matter what the condition is. This results in a better scene understanding and stable features provided by the perception module to support the controller module in estimating navigational control properly. To evaluate its performance, we conduct several tests by deploying the model to predict a set of driving records and perform real automated driving under three different conditions. We also conduct ablation and comparative studies with some recent models to justify its performance. Based on the experimental results, DeepIPCv2 shows a robust performance by achieving the best drivability in all conditions. C
    

