# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Cross-Speaker Encoding Network for Multi-Talker Speech Recognition.](http://arxiv.org/abs/2401.04152) | 本文提出了一种叫做Cross-Speaker Encoding（CSE）的网络，用于解决多说话人语音识别中的局限性，通过聚合跨说话人表示。通过与SOT结合，该模型在两个说话人的数据集上实验证明比SIMO基准模型的词错误率（WER）分别降低了8%和10%。 |
| [^2] | [RepCodec: A Speech Representation Codec for Speech Tokenization.](http://arxiv.org/abs/2309.00169) | RepCodec是一种新型的语音表示编码器，通过重构语音表示并学习矢量量化码书，将语音波形转换为语义标记。实验证明，RepCodec在语音理解和生成方面明显优于传统的k-means聚类方法。 |

# 详细

[^1]: 跨说话人编码网络用于多说话人语音识别

    Cross-Speaker Encoding Network for Multi-Talker Speech Recognition. (arXiv:2401.04152v1 [cs.SD])

    [http://arxiv.org/abs/2401.04152](http://arxiv.org/abs/2401.04152)

    本文提出了一种叫做Cross-Speaker Encoding（CSE）的网络，用于解决多说话人语音识别中的局限性，通过聚合跨说话人表示。通过与SOT结合，该模型在两个说话人的数据集上实验证明比SIMO基准模型的词错误率（WER）分别降低了8%和10%。

    

    端到端的多说话人语音识别已经引起了极大的兴趣，作为一种直接转录多个说话人重叠语音的有效方法。目前的方法通常采用1）带有分支编码器的单输入多输出（SIMO）模型，或者2）基于注意力机制的编码器-解码器架构和序列化输出训练（SOT）的单输入单输出（SISO）模型。在这项工作中，我们提出了一种叫做Cross-Speaker Encoding（CSE）的网络来解决SIMO模型的局限性，通过聚合跨说话人表示。此外，CSE模型与SOT相结合，既发挥了SIMO和SISO的优势，又缓解了它们的缺点。据我们所知，该工作代表了将SIMO和SISO集成到多说话人语音识别中的早期工作。在两个说话人的LibrispeechMix数据集上进行的实验表明，CES模型相比于SIMO基准模型将词错误率（WER）降低了8%。CSE-SOT模型将WER降低了10%

    End-to-end multi-talker speech recognition has garnered great interest as an effective approach to directly transcribe overlapped speech from multiple speakers. Current methods typically adopt either 1) single-input multiple-output (SIMO) models with a branched encoder, or 2) single-input single-output (SISO) models based on attention-based encoder-decoder architecture with serialized output training (SOT). In this work, we propose a Cross-Speaker Encoding (CSE) network to address the limitations of SIMO models by aggregating cross-speaker representations. Furthermore, the CSE model is integrated with SOT to leverage both the advantages of SIMO and SISO while mitigating their drawbacks. To the best of our knowledge, this work represents an early effort to integrate SIMO and SISO for multi-talker speech recognition. Experiments on the two-speaker LibrispeechMix dataset show that the CES model reduces word error rate (WER) by 8% over the SIMO baseline. The CSE-SOT model reduces WER by 10
    
[^2]: RepCodec:一种用于语音标记的语音表示编码器

    RepCodec: A Speech Representation Codec for Speech Tokenization. (arXiv:2309.00169v1 [eess.AS])

    [http://arxiv.org/abs/2309.00169](http://arxiv.org/abs/2309.00169)

    RepCodec是一种新型的语音表示编码器，通过重构语音表示并学习矢量量化码书，将语音波形转换为语义标记。实验证明，RepCodec在语音理解和生成方面明显优于传统的k-means聚类方法。

    

    随着大型语言模型（LLMs）的快速增长，离散语音标记在将语音注入LLMs中发挥了重要作用。然而，这种离散化导致了信息的丢失，从而损害了整体性能。为了提高这些离散语音标记的性能，我们提出了RepCodec，一种用于语义语音标记的新型语音表示编码器。与重新构建原始音频的音频编解码器不同，RepCodec通过从语音编码器（如HuBERT或data2vec）重构语音表示来学习矢量量化码书。语音编码器、编解码器和矢量量化码书共同构成一个将语音波形转换为语义标记的流水线。广泛的实验证明，由于其增强的信息保留能力，RepCodec在语音理解和生成方面显著优于广泛使用的k-means聚类方法。

    With recent rapid growth of large language models (LLMs), discrete speech tokenization has played an important role for injecting speech into LLMs. However, this discretization gives rise to a loss of information, consequently impairing overall performance. To improve the performance of these discrete speech tokens, we present RepCodec, a novel speech representation codec for semantic speech tokenization. In contrast to audio codecs which reconstruct the raw audio, RepCodec learns a vector quantization codebook through reconstructing speech representations from speech encoders like HuBERT or data2vec. Together, the speech encoder, the codec encoder and the vector quantization codebook form a pipeline for converting speech waveforms into semantic tokens. The extensive experiments illustrate that RepCodec, by virtue of its enhanced information retention capacity, significantly outperforms the widely used k-means clustering approach in both speech understanding and generation. Furthermore
    

