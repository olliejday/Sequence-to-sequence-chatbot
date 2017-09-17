#TensorFlow Seq2Seq Chatbot

#Sources:
#Largely inspired by Stanford's CS20SI Tensorflow for Deep Learning Research http://web.stanford.edu/class/cs20si/
#TensorFlow Seq2Seq Implementation https://www.tensorflow.org/tutorials/seq2seq
#Data used was my own Facebook message logs, and for development, Cornell's Movie-Dialogs Corpus https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

#==================================================================================================================================
#---------------------------------------
#TO-DO:

#try sampling / other methods to vary responses

#see Stanford chatbot assignment for ideas

###### fix sizing error in embedding call un training loop call to run_step ~ line 217
###### It seems to come from decoding outputs back to embeddings that are too large -poss due tp
###### my new evaluation print out parts -> at the moment have just done try... except

## try different hyperparameter settings etc. run longer to try and improve the model!

### RUN FROM PRE-TRAINED MODEL ON MESSAGE DATA

#----------------------------------------

# Imports
import util # Data handling helper functions
import models # Imports the model class
import settings # Contains hyper-parameters, file paths etc.
import random
import numpy as np
import tensorflow as tf
import os
import sys
import time
import argparse

#############################################################################################################
### SETUP CMD ARGS ###

parser = argparse.ArgumentParser()

parser.add_argument('--train', '-t', action='store_true', dest='train', help='Trains the model, based on prepared dataset. [True/False(default)]', default=False)

parser.add_argument('--chat', '-c', action='store_true', dest='chat', help='Test out the model with user input conversation. [True/False(default)]', default=False)

parser.add_argument('--prepare-data', '-d', action='store_true', dest='data', help='Pre-process the data before training. [True/False(default)]', default=False)

parser.add_argument('--messages-only', '-m', action='store_true', dest='messages', help='True to use messages only, false by default uses both movie-dialogs and message data. [True/False(default)]', default=False)

#############################################################################################################
### FUNCTIONS ###

def _get_buckets(messages_only):
    # Loads the data into buckets of different lenghts
    # Returns the buckets and a scale to weight bucket sampling
    test = 'test'
    train = 'train'
    if messages_only:
        test = 'messages-test'
        train = 'messages-train'
    test_buckets = util.load_data(test) # The data split into buckets by size
    train_buckets = util.load_data(train)
    train_bucket_sizes = [len(train_buckets[b]) for b in range(len(settings.BUCKETS))] # List of # samples in each bucket
    print("Buckets {}\nSizes {}".format(settings.BUCKETS, train_bucket_sizes))
    train_total_size = sum(train_bucket_sizes)
    # List of increasing numbers 0 to 1 to randomly sample a bucket weighted by its number of samples
    train_bucket_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in range(len(train_bucket_sizes))]
    print("Bucket sampling scale: {}".format(train_bucket_scale))
    return test_buckets, train_buckets, train_bucket_scale
    
def _check_restore_parameters(sess, saver):
    # Restore checkpointed parameters if any
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(settings.CKPT_PATH + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        print("\nLoading parameters from checkpoint")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('\nInitialising new parameters')
        
def _get_random_bucket(train_buckets_scale):
    # Bucket sample a random data bucket from which to choose a training example
    rand = random.random()
    return min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > rand])

def _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks):
    # Assert that all the inputs are of the expected sizing
    if len(encoder_inputs) != encoder_size:
        raise ValueError("Encoder length must be equal to the one in the bucket: {} != {}".format(len(encoder_inputs), 
                                                                                                          encoder_size))
    if len(decoder_inputs) != decoder_size:
        raise ValueError("Decoder length must be equal to the one in the bucket: {} != {}".format(len(decoder_inputs), 
                                                                                                      decoder_size))
    if len(decoder_inputs) != decoder_size:
        raise ValueError("Weights length (decoder_masks) must be equal to the one in the bucket: {} != {}".format(
                                                                                                    len(decoder_masks), 
                                                                                                    decoder_size))
                                                                                                    
def run_step(sess, 
             model, 
             encoder_inputs, 
             decoder_inputs, 
             decoder_masks, 
             bucket_id, 
             forward_only): # True for chat mode and test set evaluation, backward pass is for training
    
    encoder_size, decoder_size = settings.BUCKETS[bucket_id]
    _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks)
    
    # feed_dict: encoder inputs, decoder inputs, target weights; as provided
    feed_dict = {}
    for step in range(encoder_size):
        feed_dict[model.encoder_inputs[step].name] = encoder_inputs[step]
    for step in range(decoder_size):
        feed_dict[model.decoder_inputs[step].name] = decoder_inputs[step]
        feed_dict[model.decoder_masks[step].name] = decoder_masks[step]
    
    last_target = model.decoder_inputs[decoder_size].name
    feed_dict[last_target] = np.zeros([model.batch_size], dtype=np.int32)
    
    # ops: 
    # if forward only - just losses; if backward pass too - compute gradients and perform training step
    if forward_only:
        ops = [model.losses[bucket_id]]
        for step in range(decoder_size):
            ops.append(model.outputs[bucket_id][step]) # Output logits
    else:
        ops = [model.train_ops[bucket_id],
               model.gradient_norms[bucket_id],
               model.losses[bucket_id]]

    outputs = sess.run(ops, feed_dict=feed_dict)

    if forward_only:
        return None, outputs[0], outputs[1:] # No gradient norm, loss, outputs
    return outputs[1], outputs[2], None # Gradeint norm, loss, no outputs

def print_encoder(text, dic):
    # Takes an encoder sample of batch size 1 and prints the text form
    if settings.EOS_ID in text: # Break at EOS symbol
        text = text[:text.index(settings.EOS_ID)]
    chars = [str(dic[text[i]]) for i in range(len(text)) if text[i] > settings.EOS_ID] # Assumes EOS_ID is last special char
    chars.reverse() # Encoder inputs are reveresed
    print("Q | ", end="")
    print(" ".join(chars))
    return " ".join(chars)

def print_decoder(logits, dic, example=0):
    # Takes logits output from the model and greedily forms a response
    # TO-DO: try sampling / other methods to vary responses
    answer = [logits[char][example][:] for char in range(len(logits))]
    outputs = [int(np.argmax(ans, axis=0)) for ans in answer]
    # Break at EOS symbol
    if settings.EOS_ID in outputs:
        outputs = outputs[:outputs.index(settings.EOS_ID)]
    # Print the sentence
    print("A >> ", end="")
    chars = [str(dic[outputs[i]]) for i in range(len(outputs)) if outputs[i] > settings.EOS_ID] 
    print(" ".join(chars))
    print()
    return " ".join(chars)

def _eval_test_set(sess, model, test_buckets):
    # Evaluate on the test set
    for bucket_id in range(len(settings.BUCKETS)):
        if len(test_buckets[bucket_id]) == 0:
            print("\nEmpty test bucket {}".format(settings.BUCKETS[bucket_id]))
            continue
        
        # Run forward only on test batch
        encoder_inputs, decoder_inputs, decoder_masks = util.get_batch(test_buckets[bucket_id],
                                                                  bucket_id,
                                                                  batch_size=settings.BATCH_SIZE)#settings.BATCH_SIZE)
        _, step_loss, logits = run_step(sess, model, encoder_inputs, decoder_inputs,
                                      decoder_masks, bucket_id, True)
        
        print("\nEVALUATING ON TEST SET")
        print("\n{} | test bucket {}; test loss {}\n".format(time.strftime("%c"), settings.BUCKETS[bucket_id], step_loss))
        
        # Print random example of Q/A
        example = random.choice(range(settings.BATCH_SIZE))
        question = [encoder_inputs[char][example] for char in range(len(encoder_inputs))]

        print_encoder(question, model.encoder_to_words)
        print_decoder(logits, model.decoder_to_words, example)
            

def train(messages_only=False):
    # Trains the chatbot model defined above with the data processed above
    
    print("""

 _____         _       _             
|_   _|       (_)     (_)            
  | |_ __ __ _ _ _ __  _ _ __   __ _ 
  | | '__/ _` | | '_ \| | '_ \ / _` |
  | | | | (_| | | | | | | | | | (_| |
  \_/_|  \__,_|_|_| |_|_|_| |_|\__, |
                                __/ |
                               |___/ 

""")
    
    # Load data
    test_buckets, train_buckets, train_buckets_scale = _get_buckets(messages_only)
    # Init model
    model = models.ChatbotModel(forward_only=False, batch_size= settings.BATCH_SIZE)
    model.build_graph()
    # Init checkpoint saver
    saver = tf.train.Saver(max_to_keep=100)
    
    sess = tf.InteractiveSession() # More flexible with ipynb format
    print("\nRunning session")
    sess.run(tf.global_variables_initializer())
    _check_restore_parameters(sess, saver)
    
    iteration = model.global_step.eval()
    total_loss = 0
    
    print("\nStarting training at {}\n".format(time.strftime('%c')))
    
    for _ in range(settings.MAX_ITER): 
        bucket_id = _get_random_bucket(train_buckets_scale)
        
        encoder_inputs, decoder_inputs, decoder_masks = util.get_batch(train_buckets[bucket_id], 
                                                                  bucket_id,
                                                                  batch_size=settings.BATCH_SIZE)
        ######
        ###### Kept having errors with below line of type
        # InvalidArgumentError (see above for traceback): indices[61] = 42998 is not in [0, 42996)
        ###### SO added a try excpet wrapper so it didn't break but needs fixing
        ###### TO-DO: fix sizing error in embedding call un training loop call to run_step ~ line 217
        ###### It seems to come from decoding outputs back to embeddings that are too large -poss due tp
        ###### my new evaluation print out parts
        ######
        try:
            _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, False)
            total_loss += step_loss
            iteration += 1
        except:
            print("Error in training step []run_step()], continuing from next step")
        
        if iteration % settings.PRINT_EVERY == 0: # Print over period of iterations to reduce noise by averaging
            print("{} | Iteration {}; Loss {};".format(time.strftime('%c'), 
                                                       iteration, 
                                                       float(total_loss)/settings.PRINT_EVERY))
            total_loss = 0
        
        if iteration % settings.SAVE_EVERY == 0:
            saved_path = saver.save(sess, os.path.join(settings.CKPT_PATH, 
                                          'chatbot-ckpt-{}'.format(str(round(time.time())))),
                                          global_step=model.global_step)
            print("\nModel saved to {}".format(saved_path))
            
        if iteration % settings.EVAL_EVERY == 0:
            # run evaluation on development set and print their loss
            _eval_test_set(sess, model, test_buckets)
        
        sys.stdout.flush()

def _get_user_input():
    # Get user input which will be transformed into encoder input later
    inp = input("Q | ")
    sys.stdout.flush()
    return inp.strip()

def _find_right_bucket_length(length):
    # Find the correct bucket for an input length
    return min([b for b in range(len(settings.BUCKETS)) if settings.BUCKETS[b][0] >= length])

def chat(to_file=False):
    # Takes user input and responds with the trained model
    
    # Init model
    model = models.ChatbotModel(forward_only=True, batch_size=1)
    model.build_graph()
    # Init checkpoint saver
    saver = tf.train.Saver()
    
    sess = tf.InteractiveSession() # More flexible with ipynb format
    print("Running session")
    sess.run(tf.global_variables_initializer())
    _check_restore_parameters(sess, saver)
    
    if to_file: 
        output_file = open(os.path.join(settings.GENERATED_PATH.format(str(round(time.time())))), 'a+')
        output_file.write("="*120)
        output_file.write("{}".format(time.strftime("%c")))
    
    max_length = settings.BUCKETS[-1][0]
    
    print("="*120)
    print("""
 _____                                     _   _             
/  __ \                                   | | (_)            
| /  \/ ___  _ ____   _____ _ __ ___  __ _| |_ _  ___  _ __  
| |    / _ \| '_ \ \ / / _ \ '__/ __|/ _` | __| |/ _ \| '_ \ 
| \__/\ (_) | | | \ V /  __/ |  \__ \ (_| | |_| | (_) | | | |
 \____/\___/|_| |_|\_/ \___|_|  |___/\__,_|\__|_|\___/|_| |_|
        
    """)
    print("="*120)
    print('Welcome to Conversation.')
    print("Type up to {} chars to start, ENTER to exit.".format(max_length))
    
    while True:
        line = _get_user_input()
        
        if len(line) <= 0 or line == "": break
        
        # Tokens for input sentence
        tokens = util.embed(model.words_to_encoder, line)
        if (len(tokens) > max_length):
            print("System message: Maximum input length for this model is {}, please try again.".format(max_length))
            line = _get_user_input()
            continue
        
        bucket_id = _find_right_bucket_length(len(tokens)) # Which bucket for this input length?
        # Form the input sentence into a one element batch to feed the model
        encoder_inputs, decoder_inputs, decoder_masks = util.get_batch([(tokens, [])],
                                                                        bucket_id,
                                                                        batch_size=1)
        # Get outputs of model
        _, _, logits = run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, True)
        
        print_decoder(logits, model.decoder_to_words)
        
        if to_file:
            output_file.write("Q | " + line)
            output_file.write("A >> " + response)
    
    if to_file:
        output_file.write("="*120)
        output_file.close()

###########################################################################################################
### Run! ###
# Parse the CMD args

results = parser.parse_args()

if not results.train and not results.chat and not results.data:
    raise Warning ("Invalid arguments: no arguments found, execute with [-h] for options.")
    
if results.train and results.chat:
    raise Warning ("Training runs indefinitely, --chat will only run without --train, please identify -c or -t. [-h] for options")

print("\nRunning model with {}\n".format(results))

if results.data:
    util.prepare()

if results.train:
    train(results.messages)
    
if results.chat:
    chat()
    
