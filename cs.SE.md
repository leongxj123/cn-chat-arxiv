# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fix-Con: Automatic Fault Localization and Repair of Deep Learning Model Conversions](https://rss.arxiv.org/abs/2312.15101) | 本文提出了一种自动化的故障定位和修复方法Fix-Con，用于在深度学习模型转换过程中修复由转换引入的故障。Fix-Con能够检测和修复模型输入、参数、超参数和模型图方面的故障，提高转换模型的部署和预测正确性。 |

# 详细

[^1]: 修复-Con：深度学习模型转换的自动故障定位和修复

    Fix-Con: Automatic Fault Localization and Repair of Deep Learning Model Conversions

    [https://rss.arxiv.org/abs/2312.15101](https://rss.arxiv.org/abs/2312.15101)

    本文提出了一种自动化的故障定位和修复方法Fix-Con，用于在深度学习模型转换过程中修复由转换引入的故障。Fix-Con能够检测和修复模型输入、参数、超参数和模型图方面的故障，提高转换模型的部署和预测正确性。

    

    在不同深度学习框架之间进行模型转换是一种常见的步骤，可以最大程度地增加模型在设备之间的兼容性，并利用可能只在一个深度学习框架中提供的优化功能。然而，这个转换过程可能存在错误，导致转换后的模型无法部署或存在问题，严重降低了其预测的正确性。我们提出了一种自动化的故障定位和修复方法，Fix-Con，在深度学习框架之间进行模型转换时使用。Fix-Con能够检测和修复在转换过程中引入的模型输入、参数、超参数和模型图的故障。Fix-Con使用从调查转换问题中挖掘出的一组故障类型来定位转换模型中潜在的转换故障，并适当修复它们，例如使用源模型的参数替换目标模型的参数。这一过程在数据集中的每个图像上进行迭代执行。

    Converting deep learning models between frameworks is a common step to maximize model compatibility across devices and leverage optimization features that may be exclusively provided in one deep learning framework. However, this conversion process may be riddled with bugs, making the converted models either undeployable or problematic, considerably degrading their prediction correctness.   We propose an automated approach for fault localization and repair, Fix-Con, during model conversion between deep learning frameworks. Fix-Con is capable of detecting and fixing faults introduced in model input, parameters, hyperparameters, and the model graph during conversion.   Fix-Con uses a set of fault types mined from surveying conversion issues raised to localize potential conversion faults in the converted target model, and then repairs them appropriately, e.g. replacing the parameters of the target model with those from the source model. This is done iteratively for every image in the datas
    

