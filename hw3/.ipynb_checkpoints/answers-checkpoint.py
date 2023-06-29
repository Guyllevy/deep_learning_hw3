r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 256
    hypers['seq_len'] = 64
    hypers['h_dim'] = 1024
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.2
    hypers['learn_rate'] = 0.001
    hypers['lr_sched_factor'] = 0.3
    hypers['lr_sched_patience'] = 1
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "MASTERPIECE."
    temperature = 0.4
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**
I will present 2 reasons for spliting the corpus into sequences instead of training on the whole text:<br>
<br>
1. Batch processing: Splitting the text into sequences enables batch processing during training. By organizing the sequences into batches, it becomes possible to parallelize the computations and leverage the GPU, leading to faster training times.
<br>
2. Memory limitations: Training on the entire text as a single sequence may exceed the memory capacity of the model or the available computational resources. Breaking the corpus into smaller sequences helps mitigate these limitations and allows for more efficient training.
"""

part1_q2 = r"""
**Your answer:**
The hidden state of a GRU cell is updated based on both the current input and the previous hidden state, allowing it to retain information over longer dependencies than the sequence length.
"""

part1_q3 = r"""
**Your answer:**
The primary advantage of using an RNN is its ability to capture sequential dependencies in the data. By preserving the order of batches as we did, we ensure that the RNN processes the data in the correct sequential order, allowing it to learn and model the dependencies present in the data. Shuffling the batches would disrupt the sequential structure and make it harder for the RNN to learn meaningful patterns.
"""

part1_q4 = r"""
**Your answer:**

1. Lowering the temperature makes it so the predictions are less uniform (with more variance) and thus the model is less likely the generate a letter which the model gave a low score. in other words the model produces less of a random text.
2. when the temperature is very high the input scores to the standard softmax are of low values, thus softmax produces a more uniform probability distribution over the letters. This will affect sampling in the following way. Because the model generates letters from a more uniform distribution, letters which got a low score are more likly to be generated, thus the text will have a higher chance of spelling errors and general incoherence.
3. when the temperature is very low the input scores to the standard softmax are of high values, thus we will get a very sharp distribution of letters which will result in extremely deterministic and highly focused output. The model will tend to generate repetitive and conservative text.


"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=256,
        h_dim=256,
        z_dim=32,
        x_sigma2=0.1,
        learn_rate=0.001,
        betas=(0.9, 0.999)
    )
    
   
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
The hyperparameter x_sigma2 in a VAE controls the reconstruction variance.

Low values prioritize accurate reconstructions, resulting in sharp and detailed outputs. However, there is a risk of overfitting.
High values allow more tolerance for imperfections, leading to smoother and blurred outputs. It emphasizes the regularization component but may sacrifice reconstruction quality.

"""

part2_q2 = r"""
**Your answer:**
1.The reconstruction loss ensures accurate reconstructions, while the KL divergence loss encourages a well-structured latent space. Both components help the VAE learn to generate faithful reconstructions and capture the underlying distribution of the input data.
2.The KL divergence loss term encourages the latent-space distribution in a VAE to align with a predefined prior distribution. It promotes smoothness, captures underlying structure, and controls the capacity of the latent space.
3.KL divergence loss in a VAE leads to improved interpretability of the latent space, effective data compression, and controlled generation with regularization. These benefits enable various applications and enhance the model's capabilities in understanding and generating complex data distributions.

"""

part2_q3 = r"""
**Your answer:**
 starting by maximizing the evidence distribution allows the VAE to perform variational inference, approximate the true posterior, and balance reconstruction accuracy with latent space regularization. It enables the VAE to learn a compressed and meaningful representation of the data while generating accurate reconstructions.


"""

part2_q4 = r"""
**Your answer:**
modeling the logarithm of the latent-space variance in the VAE encoder ensures numerical stability, provides flexibility for representing a wide range of variances, and encourages exploration in the latent space.

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    hypers = dict(
        embed_dim = 48, 
        num_heads = 4,
        num_layers = 3,
        hidden_dim = 128,
        window_size = 80,
        droupout = 0.1,
        lr=0.001,
    )
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**
When using sliding-window attention, each encoder layer focuses on a specific subset of the input data within a defined window. Stacking multiple layers effectively expands the contextual view, as each subsequent layer can refer to a combination of the previous layers' sliding-windows. This is similar to how stacked Convolutional Neural Network (CNN) layers increase their receptive field, or area of attention, for the input. Therefore, stacking these layers results in a broader context in the final layer.
"""

part3_q2 = r"""
**Your answer:**
One proposed variation is "Random Sampling Attention". This involves, for each position in the sequence, attending to the fixed-size sliding window plus a fixed number of additional positions randomly sampled from the entire sequence. Attention scores are computed normally for these sampled positions while others are set to zero weight. This way, the model can consider both local (sliding window) and global (random samples) context, while keeping the computational complexity linear as in sliding-window attention.

"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


# ==============
