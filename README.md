# ChiPoGen

## Introduction
 A Chinese poems generator using GPT-based language model

 Inspired by the [nanoGPT](https://github.com/karpathy/nanoGPT) developed by Anandrej Karpathyd , the same model structure is applied for training on a Chinese Poetry dataset collected in the [Chinese Potery Dataset](https://github.com/Werneror/Poetry), and then used for Chinese potery generation.

 Some generation sample is shown in the following figure:

 ![](assests/poetry_generated_sample.PNG)

 ## Model details
The model is a standard decoder-only transformer with the following parameters

    "batch_size": 12,
    "block_size": 128,
    "vocab_size": 12966,
    "n_embd" : 384,
    "n_head" : 12,
    "n_layer": 12,
    "dropout": 0.0 