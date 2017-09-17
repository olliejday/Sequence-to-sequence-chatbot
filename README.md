# Sequence-to-sequence-chatbot

An end to end system to map input sequences to output sequences of different lengths. Here trained to develop a chatbot model. A ‘query’ is input to the GRU RNN which encodes the input to a fixed dimension vector, this is in turn input into a decoder GRU RNN which outputs the ‘response’.

Trained on Cornell Movie-Dialogs data:

![alt text](https://github.com/olliejday/Sequence-to-sequence-chatbot/blob/master/movie_chat.gif "Chatbot trained on Cornell Movie-Dialogs data")

Trained on my Facebook Message data:

![alt text](https://github.com/olliejday/Sequence-to-sequence-chatbot/blob/master/message_chat.gif "Chatbot trained on my Facebook Message data")


Implemented in Tensorflow, inspired by [2].


References:

[1] I. Sutskever, O. Vinyals, Q. Le. Sequence to Sequence Learning with Neural Networks. 2014

[2] Stanford's CS20SI Tensorflow for Deep Learning Research http://web.stanford.edu/class/cs20si/

[3] TensorFlow Seq2Seq Implementation https://www.tensorflow.org/tutorials/seq2seq

[4] Cornell's Movie-Dialogs Corpus https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
