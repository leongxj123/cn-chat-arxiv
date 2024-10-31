# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Derivative-enhanced Deep Operator Network](https://arxiv.org/abs/2402.19242) | DE-DeepONet通过整合导数信息提高了预测准确性，尤其在训练数据有限的情况下，相比传统DeepONet取得了更好的效果。 |

# 详细

[^1]: 深度导数增强的操作网络

    Derivative-enhanced Deep Operator Network

    [https://arxiv.org/abs/2402.19242](https://arxiv.org/abs/2402.19242)

    DE-DeepONet通过整合导数信息提高了预测准确性，尤其在训练数据有限的情况下，相比传统DeepONet取得了更好的效果。

    

    深度操作网络（DeepONets）是一类学习函数空间之间映射的神经算子，最近被开发为参数化偏微分方程（PDEs）的代理模型。本文提出了一种导数增强的深度操作网络（DE-DeepONet），它利用导数信息增强预测准确性，提供更准确的导数近似，特别是在训练数据有限的情况下。DE-DeepONet将输入的维度缩减到DeepONet中，并在训练的损失函数中包含两种类型的导数标签，即关于输入函数的输出函数的方向导数和关于物理域变量的输出函数的梯度。我们通过在三个不断增加复杂度的方程上测试DE-DeepONet，以展示其相对于普通DeepONet的有效性。

    arXiv:2402.19242v1 Announce Type: new  Abstract: Deep operator networks (DeepONets), a class of neural operators that learn mappings between function spaces, have recently been developed as surrogate models for parametric partial differential equations (PDEs). In this work we propose a derivative-enhanced deep operator network (DE-DeepONet), which leverages the derivative information to enhance the prediction accuracy, and provide a more accurate approximation of the derivatives, especially when the training data are limited. DE-DeepONet incorporates dimension reduction of input into DeepONet and includes two types of derivative labels in the loss function for training, that is, the directional derivatives of the output function with respect to the input function and the gradient of the output function with respect to the physical domain variables. We test DE-DeepONet on three different equations with increasing complexity to demonstrate its effectiveness compared to the vanilla DeepON
    

