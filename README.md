# name-generator

## Learn to generate names from a corpus of names with a Recurrent Neural Network.

Jupyter Notebook based on Keras and Tensorflow.

Included name corpus:
- LOTR characters from https://github.com/juandes/lotr-names-classification
- Real first names from http://www.quietaffiliate.com/free-first-name-and-last-name-databases-csv-and-sql/

Inpired by:
- https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
- DeepLearning.ai course on Sequence Models (https://www.coursera.org/learn/nlp-sequence-models)

## Approach:
1) After cleaning the name corpus from unwanted characters (titles, soft hyphens, punctuation...),
we pad all names with a starting character (#) and an ending character (*).
We use this padding characters to ease the creation of fixed-sized sequences to fit the model.

2) We then create a data dictionnary that include:
    - all names in a corpus
    - all unique characters in the corpus (including padding characters)
    - utils dictionnary to convert a given character in its corresponding index and vice versa

3) We build the input X / output Y arrays as input sequences of character (X) and next character 
after the sequence (Y). All characters are converted to a corresponding one-hot vector to ease
learning. Length of input sequence is a user choice.
    - X shape: (number_of_sequences, length_of_sequence, number_of_unique_characters_in_corpus)
    - Y shape: (number_of_sequences, number_of_unique_characters_in_corpus)
We also create some dictionnaries (trainset_utils, trainset_infos) to ease next operations.

4) We create a RNN model given trainset informations (length_of_sequence, number_of_chars) and a
number of hidden units (user choice):
    - Input layer (None, length_of_sequence, number_of_unique_characters_in_corpus)
    - LSTM layer (None, hidden_units)
    - Dense layer (None, number_of_unique_characters_in_corpus)
We also create some dictionnaries (training_infos, history) to keep traces of training perfs.

5) We compile the model with choosen hyperparams (learning rate, loss, batch size) and keep trace
of that in the history dict.

6) We train the model for a given number of epochs and keep trace of perfs in history and
training_infos dict

7) We plot some training performance graphs to see what's going on and adapt learning rate

8) Profit ! We can play with the model and generate some cool hobbit names or pseudo real
firstnames !
The name is generated from an seed sequence of padding characters (e.g. '#####' if
length_of_sequence = 5). For each step, the model take as input de previous sequence and output
propabilities for each unique characters. The next character is choosen randomly according to these
probabilities.
We also return the cumulative probabilty for the generated name (prod of each char probability) and
the cummulative gap between each choose character and the actual 'best' character (sum of gaps).
> On real first names, with a loss ~ 0.93, probability associated with a generated name seems to
accurately predict if first name exist (probability > 0.0001 means existing first name)


9) Bonus <3 We can backup/restore model with all training infos for friends or later

## Afterthoughts:

- I choose to keep upper/lower characters and special ones (e.g. 'ï', 'ê') to have more fun with
LOTR corpus but it may not be a super good idea. But the corpus seems simple enough to be ok. 
- LSTM is probably overkill for this task. Should test with GRU and SimpleRNN cell.
- With LSTM, very few hidden units seems necessary to learn names patterns. 16 hidden units with 
sequences of 5 characters give good enough results on the Hobbit Corpus
- Didn't play with batch_size, I could ?
- Didn't play with decay, I really should !
