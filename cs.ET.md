# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Quantum Circuit Fidelity Improvement with Long Short-Term Memory Networks.](http://arxiv.org/abs/2303.17523) | 本文提出使用长短期记忆网络解决量子计算中的保真度问题，利用时间序列预测方法预测量子电路的保真度。 |

# 详细

[^1]: 利用长短期记忆网络提高量子电路保真度

    Quantum Circuit Fidelity Improvement with Long Short-Term Memory Networks. (arXiv:2303.17523v1 [quant-ph])

    [http://arxiv.org/abs/2303.17523](http://arxiv.org/abs/2303.17523)

    本文提出使用长短期记忆网络解决量子计算中的保真度问题，利用时间序列预测方法预测量子电路的保真度。

    

    量子计算已进入噪声中间规模量子（NISQ）时代，目前我们拥有的量子处理器对辐射和温度等环境变量敏感，因此会产生嘈杂的输出。虽然已经有许多算法和应用程序用于NISQ处理器，但我们仍面临着解释其嘈杂结果的不确定性。具体来说，我们对所选择的量子态有多少信心？这种信心很重要，因为NISQ计算机将输出其量子位测量的概率分布，有时很难区分分布是否表示有意义的计算或只是随机噪声。本文提出了一种新方法来解决这个问题，将量子电路保真度预测框架为时间序列预测问题，因此可以利用长短期记忆（LSTM）神经网络的强大能力。一个完整的工作流程来构建训练电路

    Quantum computing has entered the Noisy Intermediate-Scale Quantum (NISQ) era. Currently, the quantum processors we have are sensitive to environmental variables like radiation and temperature, thus producing noisy outputs. Although many proposed algorithms and applications exist for NISQ processors, we still face uncertainties when interpreting their noisy results. Specifically, how much confidence do we have in the quantum states we are picking as the output? This confidence is important since a NISQ computer will output a probability distribution of its qubit measurements, and it is sometimes hard to distinguish whether the distribution represents meaningful computation or just random noise. This paper presents a novel approach to attack this problem by framing quantum circuit fidelity prediction as a Time Series Forecasting problem, therefore making it possible to utilize the power of Long Short-Term Memory (LSTM) neural networks. A complete workflow to build the training circuit d
    

