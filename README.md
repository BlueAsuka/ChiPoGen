# ChiPoGen

## Introduction
 A Chinese poems generator using GPT-based language model

 Inspired by the [nanoGPT](https://github.com/karpathy/nanoGPT) developed by Anandrej Karpathyd , the same model structure is applied for training on a Chinese Poetry dataset collected in the [Chinese Potery Dataset](https://github.com/Werneror/Poetry), and then used for Chinese potery generation.

 Some generation samples are shown in the following figure:

 ![](assests/poetry_generated_sample.PNG)

 ## Model details
The model uses a standard decoder-only transformer [GPT2](https://paperswithcode.com/paper/language-models-are-unsupervised-multitask) with the following hyperparameters. The total number of model parameters is 26.22 million.

    "batch_size": 12,
    "block_size": 128,
    "vocab_size": 12966,
    "n_embd" : 384,
    "n_head" : 12,
    "n_layer": 12
    
Note that the vocab_size of the final output layer is the total number of all chars in the Chinese potery dataset which is different from the standard implmentation of the GPT2 (vocab_size=50257). After the data processing, there are totally 12966 Chinese chars in the dataset, and this number is chosen to be the vocab_size in the output layer.

## Model training
It takes around 1 hour for finishing the model training on 5 epochs on an RTX 4070 (12G VRAM) GPU. The final evaluation loss is 3.8169.

![](assests/val_loss.png)

## Model scaling
The model can be scaled to a larger size with a larger block_size and n_embd. This will be a TODO in the future.