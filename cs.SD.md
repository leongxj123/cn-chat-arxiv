# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Daisy-TTS: Simulating Wider Spectrum of Emotions via Prosody Embedding Decomposition](https://arxiv.org/abs/2402.14523) | 本文提出了Daisy-TTS设计，通过声调嵌入分解，模拟了更广泛的情感范围，包括 primary emotions、secondary emotions、intensity-level 和 emotions polarity。 |
| [^2] | [MuTox: Universal MUltilingual Audio-based TOXicity Dataset and Zero-shot Detector.](http://arxiv.org/abs/2401.05060) | MuTox是第一个高度多语言的基于音频的毒性数据集，通过训练基于音频的毒性分类器，实现了跨多语言的零样本毒性检测，相较于现有基于文本的分类器，具有更好的性能和更广泛的语言覆盖，相较于基于词汇列表的分类器，精度和召回率提高了约2.5倍。 |

# 详细

[^1]: Daisy-TTS: 通过声调嵌入分解模拟更广泛的情感范围

    Daisy-TTS: Simulating Wider Spectrum of Emotions via Prosody Embedding Decomposition

    [https://arxiv.org/abs/2402.14523](https://arxiv.org/abs/2402.14523)

    本文提出了Daisy-TTS设计，通过声调嵌入分解，模拟了更广泛的情感范围，包括 primary emotions、secondary emotions、intensity-level 和 emotions polarity。

    

    我们经常以多方面的方式口头表达情感，它们在强度上可能有所变化，表达的不仅是单一的情感，还可能是各种情感的混合体。这种广泛的情感范围在情感结构模型中得到了深入研究，该模型将各种情感表示为原始情感的派生产品，具有不同程度的强度。在本文中，我们提出了一种情感文本转语音设计，旨在模拟基于结构模型的更广泛情感范围。我们提出的设计Daisy-TTS，结合了一个声调编码器，用于学习作为情感代理的可分离的声调嵌入。这种情感表示使模型能够模拟：（1）从训练样本中学到的原始情感，（2）作为原始情感的混合体的次级情感，（3）通过调整情感嵌入来实现强度级别，（4）通过否定情感嵌入来实现情感极性。

    arXiv:2402.14523v1 Announce Type: new  Abstract: We often verbally express emotions in a multifaceted manner, they may vary in their intensities and may be expressed not just as a single but as a mixture of emotions. This wide spectrum of emotions is well-studied in the structural model of emotions, which represents variety of emotions as derivative products of primary emotions with varying degrees of intensity. In this paper, we propose an emotional text-to-speech design to simulate a wider spectrum of emotions grounded on the structural model. Our proposed design, Daisy-TTS, incorporates a prosody encoder to learn emotionally-separable prosody embedding as a proxy for emotion. This emotion representation allows the model to simulate: (1) Primary emotions, as learned from the training samples, (2) Secondary emotions, as a mixture of primary emotions, (3) Intensity-level, by scaling the emotion embedding, and (4) Emotions polarity, by negating the emotion embedding. Through a series of
    
[^2]: MuTox: 通用多语言基于音频的毒性数据集和零样本检测器

    MuTox: Universal MUltilingual Audio-based TOXicity Dataset and Zero-shot Detector. (arXiv:2401.05060v1 [cs.SD])

    [http://arxiv.org/abs/2401.05060](http://arxiv.org/abs/2401.05060)

    MuTox是第一个高度多语言的基于音频的毒性数据集，通过训练基于音频的毒性分类器，实现了跨多语言的零样本毒性检测，相较于现有基于文本的分类器，具有更好的性能和更广泛的语言覆盖，相较于基于词汇列表的分类器，精度和召回率提高了约2.5倍。

    

    语音模态（基于音频）自然语言处理中的毒性检测研究相对有限，特别是对于非英语语言而言。为了解决这些限制，并为真正多语言的基于音频的毒性检测奠定基础，我们引入了MuTox，这是第一个具有毒性标签的高度多语言的基于音频的数据集。该数据集包含20,000个英语和西班牙语音频片段，以及其他19种语言的4,000个片段。为了证明数据集的质量，我们训练了MuTox基于音频的毒性分类器，它能够在各种语言中进行零样本毒性检测。与现有的基于文本训练的分类器相比，该分类器的AUC性能提高了超过1%，同时扩大了语言覆盖范围十倍以上。与基于词汇列表的具有相似语言覆盖数量的分类器相比，MuTox的精度和召回率提高了约2.5倍。这个显著的改进突显了其潜在的创新性和贡献。

    Research in toxicity detection in natural language processing for the speech modality (audio-based) is quite limited, particularly for languages other than English. To address these limitations and lay the groundwork for truly multilingual audio-based toxicity detection, we introduce MuTox, the first highly multilingual audio-based dataset with toxicity labels. The dataset comprises 20,000 audio utterances for English and Spanish, and 4,000 for the other 19 languages. To demonstrate the quality of this dataset, we trained the MuTox audio-based toxicity classifier, which enables zero-shot toxicity detection across a wide range of languages. This classifier outperforms existing text-based trainable classifiers by more than 1% AUC, while expanding the language coverage more than tenfold. When compared to a wordlist-based classifier that covers a similar number of languages, MuTox improves precision and recall by approximately 2.5 times. This significant improvement underscores the potenti
    

