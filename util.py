import re
import pandas as pd
import csv
import random
import numpy as np
import glob
import settings # Contains the hyper parameters, file paths etc.

#############################################################################################################
### HELPER FUNCTIONS ###

# Pass train/ test data and it will return buckets of embedded encoder and decoder data
def load_data(data):
    print("Loading ./data/{}_embed.enc".format(data))
    print("Loading ./data/{}_embed.dec".format(data))
    encoder = open("./data/{}_embed.enc".format(data))
    decoder = open("./data/{}_embed.dec".format(data))
    enc = encoder.readline()
    dec = decoder.readline()
    data_buckets = [[] for _ in settings.BUCKETS]
    while enc and dec:
        enc_ids = [int(i) for i in enc.strip().split(',') if i]
        dec_ids = [int(i) for i in dec.strip().split(',') if i]        
        for bucket_id, (enc_max, dec_max) in enumerate(settings.BUCKETS):
            if len(enc_ids) <= enc_max and len(dec_ids) <= dec_max:
                data_buckets[bucket_id].append([enc_ids, dec_ids])
                break
        enc = encoder.readline()
        dec = decoder.readline()
    return data_buckets

def pad_input(data, size):
    return data + [settings.PAD_ID] * (size - len(data))
    
def reshape_batch(data, size, batch_size):
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([data[batch_id][length_id] for batch_id in range(batch_size)], dtype=np.int32))
    return batch_inputs

def get_batch(data_buckets, bucket_id, batch_size=1):
    encoder_size, decoder_size = settings.BUCKETS[bucket_id]
    encoder_inputs = []
    decoder_inputs = []  
    for i in range(batch_size):
        encoder_input, decoder_input = random.choice(data_buckets)
        # Pad encoder and decoder, reverse the encoder
        encoder_inputs.append(list(reversed(pad_input(encoder_input, encoder_size))))
        decoder_inputs.append(pad_input(decoder_input, decoder_size))
    # Reshape into batches
    encoder_batch = reshape_batch(encoder_inputs, encoder_size, batch_size)
    decoder_batch = reshape_batch(decoder_inputs, decoder_size, batch_size)    
    # Create decoder masks
    batch_masks = []
    for length_id in range(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in range(batch_size):
            # Set mask = 0 if corresponding target is a PAD symbol
            # Corresponding decoder is decoder_input shifted forward by 1
            if length_id < decoder_size - 1:
                target = decoder_inputs[batch_id][length_id + 1]
            if length_id == decoder_size - 1 or target == settings.PAD_ID:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
    return encoder_batch, decoder_batch, batch_masks

# Split text into words, specialised for Cornell dataset
def tokenizer(in_):
    out = []
    in_ = re.sub('<u>', '', in_)
    in_ = re.sub('</u>', '', in_)
    in_ = re.sub('\[', '', in_)
    in_ = re.sub('\]', '', in_)
    in_ = re.sub("'", '', in_)
    if settings.NORMALIZE_DIGITS:
        in_ = settings.DIGIT_RE.sub('#', in_)
    for i in in_.strip().lower().split():
        out.extend(settings.WORD_SPLIT_RE.split(i))
    return [w for w in out if w]

# Create a vocab frequecy dict then write vocab to file
def build_vocab(lines, path):
    vocab = {}
    for l in lines:
        line = l[0]
        tokens = tokenizer(line)
        for w in tokens:
            word = settings.DIGIT_RE.sub("#", w) if settings.NORMALIZE_DIGITS else w
            vocab.setdefault(word, 1)
            vocab[word] += 1
    # Remove words not appearing at least twice
    [vocab.pop(k) for k,v in vocab.items() if v < 2]
    vocab_list = sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > settings.MAX_VOCAB_SIZE:
        vocab_list = vocab_list[:settings.MAX_VOCAB_SIZE]
    with open(path, "w") as vocab_file:
        vocab_file.write('<pad>' + "\n")
        vocab_file.write('<unk>' + "\n")
        vocab_file.write('<s>' + "\n")
        vocab_file.write('<\s>' + "\n")
        for w in vocab_list: 
            if vocab[w] < settings.THRESHOLD: break
            vocab_file.write(w + "\n")
    print("Created vocabulary {} with {} words\n".format(path, len(vocab_list)))
    print("{:20}\t{:20}".format("Most Common", "Least Common"))
    for i in range(20):
        print("{:20}\t{:20}".format(vocab_list[i], vocab_list[-i]))
    print()
    return len(vocab_list)
   
# Convert a line into an int embedding as defined in vocab
def embed(vocab, line):
    return [vocab.get(token, vocab['<unk>']) for token in tokenizer(line)]

# Convert the data to embedding
def embed_data(data, mode, word_embedding):
    fi = open("{}.{}".format("./data/{}".format(data), mode), "r") # in file
    fo = open("{}_embed.{}".format("./data/{}".format(data), mode), "w") # out file
    writer = csv.writer(fo)
    lines = fi.readlines()
    for l in lines:
        ids = []
        if mode == "dec": ids.append(word_embedding['<s>']) # Only care about <s> and </s> in encoder
        ids.extend(embed(word_embedding, l))
        if mode == "dec": ids.append(word_embedding['<\s>'])
        writer.writerow(ids)
    print("Embedded {}_embed.{}".format("./data/{}".format(data), mode))
  
# Embed and save all the data
def embed_dataset():
    # Load the vocab
    with open(settings.VOCAB_PATH.format("enc"), 'r') as ef:
        enc_words = ef.readlines()
    enc_embedding = {enc_words[i].strip(): i for i in range(len(enc_words))} # Dict mapping binary strings to ints

    with open(settings.VOCAB_PATH.format("dec"), 'r') as df:
        dec_words = df.readlines()
    dec_embedding = {dec_words[i].strip(): i for i in range(len(dec_words))}
    embed_data('train', 'dec', dec_embedding)
    embed_data('train', 'enc', enc_embedding)
    embed_data('test', 'dec', dec_embedding)
    embed_data('test', 'enc', enc_embedding)

def get_lines():
    # Read in the move lines
    with open(settings.DATA_PATH.format("lines"), 'r', encoding='latin-1') as f:
        movie_lines = f.readlines()
    # Get a feel for the data
    print("Read in {} movie lines\n".format(len(movie_lines)))
    for l in movie_lines[:5]:
        print(l)
    # Extract the useful dialog parts
    id_to_text = {}
    text_lines = []
    for l in movie_lines:
        l_ = l.split(settings.SPLIT)
        if l_[0] == None or l_[-1] == None or l_[0] == "" or l[-1] == "": continue
        id_to_text[l_[0]] = l_[-1].rstrip()
        text_lines.append(l_[-1].rstrip())
    print("\nSample movie line:")
    for i in id_to_text.items():
        print(i)
        break
    print()
    return id_to_text

def get_conversations():
    # Get the conversation structures
    with open(settings.DATA_PATH.format("conversations"), "r", encoding='latin-1') as f:
        conversations_data = f.readlines()
    # Extract into a list
    conv_list = [c.rstrip().split(settings.SPLIT)[-1][1:-1].split(", ") for c in conversations_data]
    conversations=[]
    for c in conv_list:
        conv = []
        for b in c:
            conv.append(b[1:-1])
        conversations.append(conv)
    # Print a sample of conversations
    print("Sample conversation structure:")
    for c in conversations[:5]:
        print(c)
    print()
    return conversations

def split_q_a():
    id_to_text = get_lines()
    conversations = get_conversations()
    # Split into Q and A
    qs, ans = [], []
    for conv in conversations:
        for i, _ in enumerate(conv[:-1]):
            if conv[i] not in id_to_text or conv[i+1] not in id_to_text: continue
            qs.append(id_to_text[conv[i]])
            ans.append(id_to_text[conv[i+1]])
    assert len(qs) == len(ans), "Q and A not the same length, {} questions and {} answers".format(len(qs), len(ans))
    print("Extracted {} questions and {} answers\n".format(len(qs), len(ans)))
    # Sample some dialogues
    for i in random.sample(range(len(qs)), 5):
        print("Q: {}".format(qs[i]))
        print("A: {}".format(ans[i]))
    print()
    return qs, ans
    
# Updates parameters in settings.py dynamically based on the dataset using file i/o -> CAREFUL not to upset settings.py with this!
# Mainly for use with ENC_ and DEC_ VOCAB_LEN
def update_settings(param, set_to):
    f = open("settings.py", "r+")
    lines = f.readlines()
    new_lines = [l for l in lines] # copy for update list
    for i, line in enumerate(lines):
        if param in line: # If already set in text, update it
            new_lines[i] = "\n{} = {}\n".format(param, set_to)
            f.close()
            f = open("settings.py", "w")
            f.writelines(new_lines)
            return
    # Otherwise add to bottom
    f.write("\n{} = {}\n".format(param, set_to))
    f.close()

def prepare_data(qs, ans):
    # Split randomly into train and test
    dl = len(qs) # How much data we're working with
    train_len = int(dl * settings.TRAIN_PCT)
    train_ids = random.sample(range(dl), train_len)
    # Q and A into pandas format
    qs_df = pd.DataFrame(qs)
    ans_df = pd.DataFrame(ans)
    train_q = qs_df.loc[train_ids]
    train_a = ans_df.loc[train_ids]
    test_q = qs_df.drop(train_ids)
    test_a = ans_df.drop(train_ids)
    print("\nTraining with {} questions and {} answers".format(len(train_q), len(train_a)))
    print("Testing with {} questions and {} answers\n".format(len(test_q), len(test_a)))
    # Write to files
    train_q.to_csv("./data/train.enc", index=False, header=False)
    train_a.to_csv("./data/train.dec", index=False, header=False)
    test_q.to_csv("./data/test.enc", index=False, header=False)
    test_a.to_csv("./data/test.dec", index=False, header=False)
    # Make vocab files based on training data
    ENC_VOCAB_LEN = build_vocab(train_q.values, settings.VOCAB_PATH.format('enc'))
    DEC_VOCAB_LEN = build_vocab(train_a.values, settings.VOCAB_PATH.format('dec'))
    # Update settings file with the correct params
    update_settings("ENC_VOCAB_LEN", ENC_VOCAB_LEN)
    update_settings("DEC_VOCAB_LEN", DEC_VOCAB_LEN)
    embed_dataset()

def load_vocab(typ):
    # Load the saved vocab file into a list and dict
    with open(settings.VOCAB_PATH.format(typ), 'r', encoding='utf-8') as f:
        words = f.read().splitlines()
    return {i: words[i] for i in range(len(words))}, {words[i]: i for i in range(len(words))}

# Above functions wrapped in message class to apply to facebook message data with different settings and file paths
# Note vocab used is vocab.enc and vocab.dec, not a unique message vocab
class MessagePrepare:
    def __init__(self):
        self.file_list = glob.glob("./data/messages/*.txt")
        self.DATA_PATH = "./data/messages-{}.{}"
        self.VOCAB_PATH = "./data/vocab.{}" # Same as pre-train/ full vocab
        # Hyperparams
        self.TRAIN_PCT = 0.85
        self.DIGIT_RE = re.compile("\d")
        self.WORD_SPLIT_RE = re.compile("([.,!?\"-<>:;)(])")
        self.NORMALIZE_DIGITS = True
        self.MAX_VOCAB_SIZE = 1e5
        self.BUCKETS = [(8, 10), (12, 14), (16, 19)]
        # Special char ids
        self.PAD_ID = 0
        self.UNK_ID = 1
        self.START_ID = 2
        self.EOS_ID = 3
        self.THRESHOLD = 2 # Minimum number of occurences for vocab list
    
    def split_q_a(self):
        qs = [] # questions (encoder)
        ans = [] # answers (decoder)
        for file in self.file_list:
            with open(file) as f:
                messages_raw = f.readlines()

            messages_raw.reverse() # Files are backwards chronologically
            # Split each line into ['name','message]
            messages = [[m.strip() for m in msg.split("<split>")] for msg in messages_raw]
            # Iterate through message list, naively taking any series of the form:
            #   Message by someone else -> message by me
            # To be a question and answer as data for the chatbot
            for i in range(len(messages) - 1):
                if messages[i][0] != 'Ollie Day' and messages[i + 1][0] == 'Ollie Day' and len(messages[i]) == 2:
                    # Add the messages to the lists, encode as ascii and decode to remove eg. emojis
                    qs.append(messages[i][1].encode('ascii', errors='ignore').decode()) # What other posted = qs
                    ans.append(messages[i + 1][1].encode('ascii', errors='ignore').decode()) # My reply = ans
        # Print a sample
        for i in random.sample(range(len(qs)), 10):
            print("\nQ | {}".format(qs[i]))
            print("A >> {}".format(ans[i]))
            
        return qs, ans
    
    # Split text into words, specialised for Cornell dataset
    def tokenizer(self, in_):
        out = []
        in_ = re.sub('<u>', '', in_)
        in_ = re.sub('</u>', '', in_)
        in_ = re.sub('\[', '', in_)
        in_ = re.sub('\]', '', in_)
        in_ = re.sub("'", '', in_)
        if self.NORMALIZE_DIGITS:
            in_ = self.DIGIT_RE.sub('#', in_)
        for i in in_.strip().lower().split():
            out.extend(self.WORD_SPLIT_RE.split(i))
        return [w for w in out if w]
    
    # Convert a line into an int embedding as defined in vocab
    def embed(self, vocab, line):
        return [vocab.get(token, vocab['<unk>']) for token in self.tokenizer(line)]
    
    # Convert the data to embedding
    def embed_data(self, data, mode, word_embedding):
        fi = open(self.DATA_PATH.format(data, mode), "r") # in file
        fo = open(self.DATA_PATH.format(data + "_embed", mode), "w") # out file
        writer = csv.writer(fo)
        lines = fi.readlines()
        for l in lines:
            ids = []
            if mode == "dec": ids.append(word_embedding['<s>']) # Only care about <s> and </s> in encoder
            ids.extend(embed(word_embedding, l))
            if mode == "dec": ids.append(word_embedding['<\s>'])
            writer.writerow(ids)
        print("Embedded {}".format(self.DATA_PATH.format(data + "_embed", mode)))
    
    # Embed and save all the data
    def embed_dataset(self):
        # Load the vocab
        with open(self.VOCAB_PATH.format("enc"), 'r') as ef:
            enc_words = ef.readlines()
        enc_embedding = {enc_words[i].strip(): i for i in range(len(enc_words))} # Dict mapping binary strings to ints

        with open(self.VOCAB_PATH.format("dec"), 'r') as df:
            dec_words = df.readlines()
        dec_embedding = {dec_words[i].strip(): i for i in range(len(dec_words))}
        self.embed_data('train', 'dec', dec_embedding)
        self.embed_data('train', 'enc', enc_embedding)
        self.embed_data('test', 'dec', dec_embedding)
        self.embed_data('test', 'enc', enc_embedding)
        
    def prepare_data(self, qs, ans):
        qa, ans = self.split_q_a()
        # Split randomly into train and test
        dl = len(qs) # How much data we're working with
        train_len = int(dl * self.TRAIN_PCT)
        train_ids = random.sample(range(dl), train_len)
        # Q and A into pandas format
        qs_df = pd.DataFrame(qs)
        ans_df = pd.DataFrame(ans)
        train_q = qs_df.loc[train_ids]
        train_a = ans_df.loc[train_ids]
        test_q = qs_df.drop(train_ids)
        test_a = ans_df.drop(train_ids)
        print("\nTraining messages with {} questions and {} answers".format(len(train_q), len(train_a)))
        print("Testing messages with {} questions and {} answers\n".format(len(test_q), len(test_a)))
        # Write to files
        train_q.to_csv(self.DATA_PATH.format("train", "enc"), index=False, header=False)
        train_a.to_csv(self.DATA_PATH.format("train", "dec"), index=False, header=False)
        test_q.to_csv(self.DATA_PATH.format("test", "enc"), index=False, header=False)
        test_a.to_csv(self.DATA_PATH.format("test", "dec"), index=False, header=False)
        self.embed_dataset()
        
    # Load the saved vocab file into a list and dict
    def load_vocab(self, typ):
        with open(self.VOCAB_PATH.format(typ), 'r', encoding='utf-8') as f:
            words = f.read().splitlines()
        return [words[i] for i in range(len(words))]

# Call other functions and methods to prepare message and pre-train datasets
# NOTE: Use the vocab.enc and vocab.dec NOT unique message vocab as they need to share vocabs
def prepare():
    print("""

______               ______                            _             
| ___ \              | ___ \                          (_)            
| |_/ / __ ___ ______| |_/ / __ ___   ___ ___  ___ ___ _ _ __   __ _ 
|  __/ '__/ _ \______|  __/ '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
| |  | | |  __/      | |  | | | (_) | (_|  __/\__ \__ \ | | | | (_| |
\_|  |_|  \___|      \_|  |_|  \___/ \___\___||___/___/_|_| |_|\__, |
                                                                __/ |
                                                               |___/ 

    """)
    # Extract and merge qs and ans
    qs, ans = split_q_a()
    msg_prep = MessagePrepare()
    msg_qs, msg_ans = msg_prep.split_q_a()
    qs.extend(msg_qs)
    ans.extend(msg_ans)

    # Prepare joint data, so that training with movie dataset mostly (as it's much larger it will dominate)
    # This will act as pre training but needs the message data too to have the correct vocabs
    prepare_data(qs, ans)

    # Prepare message only data to focus on the personalised chatbot
    msg_prep.prepare_data(msg_qs, msg_ans)
