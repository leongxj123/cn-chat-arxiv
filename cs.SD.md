# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Gaussian Adaptive Attention is All You Need: Robust Contextual Representations Across Multiple Modalities.](http://arxiv.org/abs/2401.11143) | 该论文提出了一个名为GAAM的多头高斯自适应注意力机制，用于增强跨多个模态的信息聚合。通过将可学习的均值和方差纳入注意力机制中，GAAM能够动态地重新调整特征的重要性，从而在处理非平稳数据时取得了显著的性能提升，超过了目前现有的注意力技术。该方法的适应性强且参数数量较少，具有改进现有注意力框架的潜力。 |
| [^2] | [Harnessing the Zero-Shot Power of Instruction-Tuned Large Language Model in End-to-End Speech Recognition.](http://arxiv.org/abs/2309.10524) | 本论文结合指导调整的大语言模型（LLM）和端到端自动语音识别（ASR），利用LLM的零-shot能力来改善语音识别性能。 |
| [^3] | [Whisper-KDQ: A Lightweight Whisper via Guided Knowledge Distillation and Quantization for Efficient ASR.](http://arxiv.org/abs/2305.10788) | 本文提出了一种通过引导知识蒸馏和量化，实现对大型预训练语音识别模型Whisper进行压缩优化的方法，可以将模型大小缩小并提高性能。 |

# 详细

[^1]: 高斯自适应注意力是唯一所需的：跨多个模态的健壮上下文表示

    Gaussian Adaptive Attention is All You Need: Robust Contextual Representations Across Multiple Modalities. (arXiv:2401.11143v1 [cs.LG])

    [http://arxiv.org/abs/2401.11143](http://arxiv.org/abs/2401.11143)

    该论文提出了一个名为GAAM的多头高斯自适应注意力机制，用于增强跨多个模态的信息聚合。通过将可学习的均值和方差纳入注意力机制中，GAAM能够动态地重新调整特征的重要性，从而在处理非平稳数据时取得了显著的性能提升，超过了目前现有的注意力技术。该方法的适应性强且参数数量较少，具有改进现有注意力框架的潜力。

    

    我们提出了多头高斯自适应注意力机制（GAAM），一种新颖的概率注意力框架，并设计了高斯自适应变压器（GAT），旨在增强跨多个模态（包括语音、文本和视觉）的信息聚合。GAAM将可学习的均值和方差融入其注意力机制中，采用多头框架实现，使其能够集体建模任何概率分布，以动态重新调整特征重要性。该方法在处理高度非平稳数据时表现出显著改进，通过识别特征空间中的关键元素，超越了现有的注意力技术在模型性能上的状态（精度增加约20%）。GAAM与基于点积的注意力模型兼容，并具有相对较低的参数数量，展示了其适应性和提升现有注意力框架的潜力。在实证方面，GAAM表现出卓越的适应性和功效。

    We propose the Multi-Head Gaussian Adaptive Attention Mechanism (GAAM), a novel probabilistic attention framework, and the Gaussian Adaptive Transformer (GAT), designed to enhance information aggregation across multiple modalities, including Speech, Text and Vision. GAAM integrates learnable mean and variance into its attention mechanism, implemented in a Multi-Headed framework enabling it to collectively model any Probability Distribution for dynamic recalibration of feature significance. This method demonstrates significant improvements, especially with highly non-stationary data, surpassing the state-of-the-art attention techniques in model performance (up to approximately +20% in accuracy) by identifying key elements within the feature space. GAAM's compatibility with dot-product-based attention models and relatively low number of parameters showcases its adaptability and potential to boost existing attention frameworks. Empirically, GAAM exhibits superior adaptability and efficacy
    
[^2]: 发挥指导调整的大语言模型在端到端语音识别中的零-shot能力

    Harnessing the Zero-Shot Power of Instruction-Tuned Large Language Model in End-to-End Speech Recognition. (arXiv:2309.10524v1 [eess.AS])

    [http://arxiv.org/abs/2309.10524](http://arxiv.org/abs/2309.10524)

    本论文结合指导调整的大语言模型（LLM）和端到端自动语音识别（ASR），利用LLM的零-shot能力来改善语音识别性能。

    

    我们提出了一种将指导调整的大语言模型和端到端自动语音识别相结合的新方法。现代大语言模型在零-shot学习中可以执行各种语言任务，只要提供明确的指导或提示来指导文本生成过程。我们探索使用这种零-shot能力的大语言模型来提取语言信息，以改善语音识别性能。具体来说，我们将大语言模型引导去纠正语音识别假设中的语法错误，并利用嵌入的语言知识进行端到端语音识别。所提出的模型基于混合连接主义时间分类和注意力架构，其中指导调整的大语言模型（即Llama2）被用作解码器的前端。通过CTC解码从编码器获得一个需要纠正的语音识别假设，然后将其与指导一起输入大语言模型。解码器随后采取...

    We present a novel integration of an instruction-tuned large language model (LLM) and end-to-end automatic speech recognition (ASR). Modern LLMs can perform a wide range of linguistic tasks within zero-shot learning when provided with a precise instruction or a prompt to guide the text generation process towards the desired task. We explore using this zero-shot capability of LLMs to extract linguistic information that can contribute to improving ASR performance. Specifically, we direct an LLM to correct grammatical errors in an ASR hypothesis and harness the embedded linguistic knowledge to conduct end-to-end ASR. The proposed model is built on the hybrid connectionist temporal classification (CTC) and attention architecture, where an instruction-tuned LLM (i.e., Llama2) is employed as a front-end of the decoder. An ASR hypothesis, subject to correction, is obtained from the encoder via CTC decoding, which is then fed into the LLM along with an instruction. The decoder subsequently tak
    
[^3]: Whisper-KDQ: 通过引导知识蒸馏和量化实现高效ASR的轻型Whisper

    Whisper-KDQ: A Lightweight Whisper via Guided Knowledge Distillation and Quantization for Efficient ASR. (arXiv:2305.10788v1 [cs.SD])

    [http://arxiv.org/abs/2305.10788](http://arxiv.org/abs/2305.10788)

    本文提出了一种通过引导知识蒸馏和量化，实现对大型预训练语音识别模型Whisper进行压缩优化的方法，可以将模型大小缩小并提高性能。

    

    随着计算硬件资源的快速发展和数据的显著增长，预训练模型在语音识别等任务中的应用显著提高了性能。然而，这些模型通常具有很高的计算开销，使其难以在资源受限的设备上有效执行。为了加速推理、减少模型大小，并保持性能，我们提出了一种新颖的引导知识蒸馏和量化方法，用于大型预训练模型Whisper。学生模型基于量化损失和蒸馏损失选择蒸馏和量化层。我们将$\text{Whisper}_\text{small}$压缩到$\text{Whisper}_\text{base}$和$\text{Whisper}_\text{tiny}$级别，使$\text{Whisper}_\text{small}$分别小5.18x/10.48x。此外，与原始$\text{Whisper}_\text{base}$和$\text{Whisper}_\text{tiny}$相比，还有相对字符错误率降低.

    Due to the rapid development of computing hardware resources and the dramatic growth of data, pre-trained models in speech recognition, such as Whisper, have significantly improved the performance of speech recognition tasks. However, these models usually have a high computational overhead, making it difficult to execute effectively on resource-constrained devices. To speed up inference and reduce model size while maintaining performance, we propose a novel guided knowledge distillation and quantization for large pre-trained model Whisper. The student model selects distillation and quantization layers based on quantization loss and distillation loss, respectively. We compressed $\text{Whisper}_\text{small}$ to $\text{Whisper}_\text{base}$ and $\text{Whisper}_\text{tiny}$ levels, making $\text{Whisper}_\text{small}$ 5.18x/10.48x smaller, respectively. Moreover, compared to the original $\text{Whisper}_\text{base}$ and $\text{Whisper}_\text{tiny}$, there is also a relative character erro
    

