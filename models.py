import util
import settings
import tensorflow as tf

class ChatbotModel(object):
    def __init__(self, forward_only, batch_size):
        print("\nInitialising new model.")
        self.forward_only = forward_only
        self.batch_size = batch_size
        self.encoder_to_words, self.words_to_encoder = util.load_vocab('enc')
        self.decoder_to_words, self.words_to_decoder = util.load_vocab('dec')
    
    def _create_placeholders(self):
        print("Creating placeholder.")
        # Bucket for each int range(0 to largest bucker) - last bucket is largest
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name="enc_in{}".format(i)) 
                          for i in range(settings.BUCKETS[-1][0])]
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name="dec_in{}".format(i)) 
                          for i in range(settings.BUCKETS[-1][1] + 1)]
        self.decoder_masks = [tf.placeholder(tf.float32, shape=[None], name="dec_mask{}".format(i)) 
                         for i in range(settings.BUCKETS[-1][1] + 1)]
        # Targets are decoder inputs shifted by one (to ignore <s> symbol)
        self.targets = self.decoder_inputs[1:]
    
    def _inference(self):
        print("Defining inference.")
        # Sampled softmax only makes sense if sample less than vocab size (ie. NUM_SAMPLES < settings.DEC_VOCAB_LEN)
        if settings.NUM_SAMPLES > 0 and settings.NUM_SAMPLES< settings.DEC_VOCAB_LEN:
            w_t = tf.get_variable('projection_w', [settings.DEC_VOCAB_LEN, settings.HIDDEN_SIZE], dtype=tf.float32)
            w = tf.transpose(w_t)
            b = tf.get_variable('projection_b', [settings.DEC_VOCAB_LEN], dtype=tf.float32)
            self.output_projection = (w, b)

        ########                                                                     #########
        ######## HAD SOME PROBLEMS WITH SAMPLED_LOSS_FUNCTION SO COME BELOW TO DEBUG #########
        ########                                                                     #########
        
        def sampled_loss_function(labels, logits):
            labels = tf.reshape(labels, [-1, 1])
            # Compute sampled_softmax_loss using 32-bit floats for numerical stability
            local_inputs = tf.cast(logits, tf.float32)
            return tf.nn.sampled_softmax_loss(
                                    weights=w_t,
                                    biases=b,
                                    labels=labels,
                                    inputs=local_inputs,
                                    num_sampled=settings.NUM_SAMPLES,
                                    num_classes=settings.DEC_VOCAB_LEN)
                                            
        self.softmax_loss_function = sampled_loss_function
        
        def single_cell():
            return tf.nn.rnn_cell.GRUCell(settings.HIDDEN_SIZE)
        self.enc_cell = single_cell()
        self.dec_cell = single_cell()
        if settings.NUM_LAYERS > 1:
            self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell() for _ in range(settings.NUM_LAYERS)])
            self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell() for _ in range(settings.NUM_LAYERS)])
    
    def _create_loss(self):
        print("Creating loss.")
        def seq2seq_function(enc_inputs, dec_inputs, do_decode):
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                        encoder_inputs=enc_inputs, decoder_inputs=dec_inputs, 
                        enc_cell=self.enc_cell, dec_cell=self.dec_cell, 
                        num_encoder_symbols=settings.ENC_VOCAB_LEN,
                        num_decoder_symbols=settings.DEC_VOCAB_LEN,
                        embedding_size=settings.HIDDEN_SIZE,
                        output_projection=self.output_projection,
                        feed_previous=do_decode)
        
        if self.forward_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                                            self.encoder_inputs,
                                            self.decoder_inputs,
                                            self.targets,
                                            self.decoder_masks,
                                            settings.BUCKETS,
                                            lambda x, y: seq2seq_function(x, y, True),
                                            softmax_loss_function=self.softmax_loss_function)
            # If using output projection, need to project for decoding
            if self.output_projection:
                for bucket in range(len(settings.BUCKETS)):
                    self.outputs[bucket] = [tf.matmul(output, self.output_projection[0]) + self.output_projection[1]
                                               for output in self.outputs[bucket]]
            
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets( 
                                            self.encoder_inputs,
                                            self.decoder_inputs,
                                            self.targets,
                                            self.decoder_masks,
                                            settings.BUCKETS,
                                            lambda x, y: seq2seq_function(x, y, False),
                                            softmax_loss_function=self.softmax_loss_function)
            
            # If using output projection, need to project to get full decoding, this is used in printing evaluation
            if self.output_projection:
                for bucket in range(len(settings.BUCKETS)):
                    self.outputs[bucket] = [tf.matmul(output, self.output_projection[0]) + self.output_projection[1]
                                               for output in self.outputs[bucket]]
                                            
                                            
    def _create_optimizer(self):
        print("Creating optimizer.")
        with tf.variable_scope('training') as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            
            if not self.forward_only:
                self.optimizer = tf.train.GradientDescentOptimizer(settings.LEARNING_RATE) # WAS GRAD DESC IN STANFORD CODE
                trainables = tf.trainable_variables()
                self.gradient_norms = []
                self.train_ops = []
                for bucket in range(len(settings.BUCKETS)):
                    clipped_gradients, norm = tf.clip_by_global_norm(tf.gradients(self.losses[bucket], 
                                                                                 trainables), 
                                                                                 settings.MAX_GRADIENT_NORM)
                    self.gradient_norms.append(norm)
                    self.train_ops.append(self.optimizer.apply_gradients(zip(clipped_gradients, trainables),
                                                              global_step=self.global_step))
    
    def _create_summaries(self):
        # Summaries placeholder: for plots and graph def etc. for tensorboard
        return
    
    def build_graph(self):
        self._create_placeholders()
        self._inference()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()
