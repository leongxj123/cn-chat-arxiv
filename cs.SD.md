# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fast Word Error Rate Estimation Using Self-Supervised Representations For Speech And Text.](http://arxiv.org/abs/2310.08225) | 本文介绍了一种使用自监督学习表示法（SSLR）的快速WER估计器（Fe-WER），在大数据场景下具有较高的计算效率和性能提升。 |

# 详细

[^1]: 使用自监督表示法对语音和文本进行快速字错率估计

    Fast Word Error Rate Estimation Using Self-Supervised Representations For Speech And Text. (arXiv:2310.08225v1 [eess.AS])

    [http://arxiv.org/abs/2310.08225](http://arxiv.org/abs/2310.08225)

    本文介绍了一种使用自监督学习表示法（SSLR）的快速WER估计器（Fe-WER），在大数据场景下具有较高的计算效率和性能提升。

    

    自动语音识别（ASR）的质量通常通过字错率（WER）来衡量。WER估计是一项任务，旨在预测ASR系统的WER，给定一个语音说话和一个转录。在大量数据上训练先进的ASR系统的同时，这个任务越来越受到关注。在这种情况下，WER估计在许多场景中变得必要，例如选择具有未知转录质量的训练数据，或在没有地面真实转录的情况下估计ASR系统的测试性能。面对大量数据，WER估计仪的运算效率在实际应用中变得至关重要。然而，以前的研究通常未将其视为优先考虑的问题。本文介绍了一种使用自监督学习表示法（SSLR）的快速WER估计器（Fe-WER）。该估计器基于通过平均池聚合的SSLR构建。结果表明，相对于e-WER3基线，Fe-WER的性能提高了19.69％。

    The quality of automatic speech recognition (ASR) is typically measured by word error rate (WER). WER estimation is a task aiming to predict the WER of an ASR system, given a speech utterance and a transcription. This task has gained increasing attention while advanced ASR systems are trained on large amounts of data. In this case, WER estimation becomes necessary in many scenarios, for example, selecting training data with unknown transcription quality or estimating the testing performance of an ASR system without ground truth transcriptions. Facing large amounts of data, the computation efficiency of a WER estimator becomes essential in practical applications. However, previous works usually did not consider it as a priority. In this paper, a Fast WER estimator (Fe-WER) using self-supervised learning representation (SSLR) is introduced. The estimator is built upon SSLR aggregated by average pooling. The results show that Fe-WER outperformed the e-WER3 baseline relatively by 19.69% an
    

