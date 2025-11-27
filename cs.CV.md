# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LTD: Low Temperature Distillation for Robust Adversarial Training.](http://arxiv.org/abs/2111.02331) | 本文提出了一种名为低温蒸馏（LTD）的新方法，通过使用修改的知识蒸馏框架生成软标签，解决了对抗训练中常用的独热向量标签带来的学习困难问题，提高了模型的稳健性。 |

# 详细

[^1]: 低温蒸馏：用于稳健对抗训练的方法

    LTD: Low Temperature Distillation for Robust Adversarial Training. (arXiv:2111.02331v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2111.02331](http://arxiv.org/abs/2111.02331)

    本文提出了一种名为低温蒸馏（LTD）的新方法，通过使用修改的知识蒸馏框架生成软标签，解决了对抗训练中常用的独热向量标签带来的学习困难问题，提高了模型的稳健性。

    

    对抗训练已经被广泛应用于增强神经网络模型对抗攻击的稳健性。尽管神经网络模型很受欢迎，但是这些模型的自然准确性和稳健准确性之间存在着显著差距。本文的主要贡献是发现了这个差距的一个主要原因是常用的独热向量作为标签，这阻碍了图像识别的学习过程。用独热向量表示模糊图像是不准确的，可能导致模型得到次优解。为了解决这个问题，我们提出了一种新颖的方法，称之为低温蒸馏（LTD），它使用修改的知识蒸馏框架生成软标签。与以前的方法不同，LTD在教师模型中使用相对较低的温度，而对教师和学生模型使用固定但不同的温度。这个修改可以提高模型的稳健性，而不会遇到已经在先前工作中解决的梯度掩码问题。

    Adversarial training has been widely used to enhance the robustness of neural network models against adversarial attacks. Despite the popularity of neural network models, a significant gap exists between the natural and robust accuracy of these models. In this paper, we identify one of the primary reasons for this gap is the common use of one-hot vectors as labels, which hinders the learning process for image recognition. Representing ambiguous images with one-hot vectors is imprecise and may lead the model to suboptimal solutions. To overcome this issue, we propose a novel method called Low Temperature Distillation (LTD) that generates soft labels using the modified knowledge distillation framework. Unlike previous approaches, LTD uses a relatively low temperature in the teacher model and fixed, but different temperatures for the teacher and student models. This modification boosts the model's robustness without encountering the gradient masking problem that has been addressed in defe
    

