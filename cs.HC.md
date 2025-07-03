# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diffusion-based Iterative Counterfactual Explanations for Fetal Ultrasound Image Quality Assessment](https://arxiv.org/abs/2403.08700) | 提出了基于扩散的迭代反事实解释的方法，通过生成逼真的高质量标准平面，对提高临床医生的培训、改善图像质量以及提升下游诊断和监测具有潜在价值。 |
| [^2] | [Feature Reweighting for EEG-based Motor Imagery Classification.](http://arxiv.org/abs/2308.02515) | 本论文提出了一种特征重加权的方法，用于解决使用EEG信号进行运动想象分类时存在的低信噪比、非稳态性、非线性和复杂性等挑战，通过降低噪声和无关信息，提高分类性能。 |

# 详细

[^1]: 基于扩散的迭代反事实解释用于胎儿超声图像质量评估

    Diffusion-based Iterative Counterfactual Explanations for Fetal Ultrasound Image Quality Assessment

    [https://arxiv.org/abs/2403.08700](https://arxiv.org/abs/2403.08700)

    提出了基于扩散的迭代反事实解释的方法，通过生成逼真的高质量标准平面，对提高临床医生的培训、改善图像质量以及提升下游诊断和监测具有潜在价值。

    

    怀孕期超声图像质量对准确诊断和监测胎儿健康至关重要。然而，生成高质量的标准平面很困难，受到超声波技术人员的专业知识以及像孕妇BMI或胎儿动态等因素的影响。在这项工作中，我们提出使用基于扩散的反事实可解释人工智能，从低质量的非标准平面生成逼真的高质量标准平面。通过定量和定性评估，我们证明了我们的方法在生成质量增加的可信反事实方面的有效性。这为通过提供视觉反馈加强临床医生培训以及改进图像质量，从而改善下游诊断和监测提供了未来的希望。

    arXiv:2403.08700v1 Announce Type: cross  Abstract: Obstetric ultrasound image quality is crucial for accurate diagnosis and monitoring of fetal health. However, producing high-quality standard planes is difficult, influenced by the sonographer's expertise and factors like the maternal BMI or the fetus dynamics. In this work, we propose using diffusion-based counterfactual explainable AI to generate realistic high-quality standard planes from low-quality non-standard ones. Through quantitative and qualitative evaluation, we demonstrate the effectiveness of our method in producing plausible counterfactuals of increased quality. This shows future promise both for enhancing training of clinicians by providing visual feedback, as well as for improving image quality and, consequently, downstream diagnosis and monitoring.
    
[^2]: 基于EEG的运动想象分类的特征重加权

    Feature Reweighting for EEG-based Motor Imagery Classification. (arXiv:2308.02515v1 [cs.LG])

    [http://arxiv.org/abs/2308.02515](http://arxiv.org/abs/2308.02515)

    本论文提出了一种特征重加权的方法，用于解决使用EEG信号进行运动想象分类时存在的低信噪比、非稳态性、非线性和复杂性等挑战，通过降低噪声和无关信息，提高分类性能。

    

    利用非侵入性脑电图（EEG）信号进行运动想象（MI）分类是一个重要的目标，因为它用于预测主体肢体移动的意图。最近的研究中，基于卷积神经网络（CNN）的方法已被广泛应用于MI-EEG分类。训练神经网络进行MI-EEG信号分类的挑战包括信噪比低、非稳态性、非线性和EEG信号的复杂性。基于CNN的网络计算得到的MI-EEG信号特征包含无关信息。因此，由噪声和无关特征计算得到的CNN网络的特征图也包含无关信息。因此，许多无用的特征常常误导神经网络训练，降低分类性能。为解决这个问题，提出了一种新的特征重加权方法。

    Classification of motor imagery (MI) using non-invasive electroencephalographic (EEG) signals is a critical objective as it is used to predict the intention of limb movements of a subject. In recent research, convolutional neural network (CNN) based methods have been widely utilized for MI-EEG classification. The challenges of training neural networks for MI-EEG signals classification include low signal-to-noise ratio, non-stationarity, non-linearity, and high complexity of EEG signals. The features computed by CNN-based networks on the highly noisy MI-EEG signals contain irrelevant information. Subsequently, the feature maps of the CNN-based network computed from the noisy and irrelevant features contain irrelevant information. Thus, many non-contributing features often mislead the neural network training and degrade the classification performance. Hence, a novel feature reweighting approach is proposed to address this issue. The proposed method gives a noise reduction mechanism named
    

