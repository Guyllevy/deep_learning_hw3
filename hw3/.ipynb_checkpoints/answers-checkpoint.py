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
    temperature = 0.5
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
   
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


"""

part2_q2 = r"""
**Your answer:**


"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**


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
    
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**


"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


# ==============
