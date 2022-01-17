import numpy as np
from gensim.models import KeyedVectors


def getVectors(args, wordvocab):
    vectors = []
    hit=0
    if args.mode != 'rand':
        word2vec = KeyedVectors.load_word2vec_format('../../lin_absa/yelp_code/GoogleNews-vectors-negative300.bin', binary=True)
        for i in range(len(wordvocab)):
            word = wordvocab[i]
            if word in word2vec.vocab:
                vectors.append(word2vec[word])
                hit +=1
            else:
                vectors.append(np.random.uniform(-0.01, 0.01, args.embed_dim))
    else:
        for i in range(len(wordvocab)):
            vectors.append(np.random.uniform(-0.01, 0.01, args.embed_dim))
    hit_rate = float(hit)/len(wordvocab)
    print(("The hit rate is {}".format(hit_rate)))
    return np.array(vectors)

def getVectors2(embed_dim,wordvocab):
    vectors = []
    id2word = {}
    hit = 0
    for word in wordvocab.keys():
        id2word[wordvocab[word]]=word
    # id2word = wordvocab
    if id2word != None:
        word2vec = KeyedVectors.load_word2vec_format('/mnt/sda/media/Data2/hanqi/sine/GoogleNews-vectors-negative300.bin', binary=True)
        for i in range(len(id2word)):
            word = id2word[i]
            if word in word2vec.key_to_index.keys():
                vectors.append(word2vec[word])
                hit +=1
            else:
                vectors.append(np.random.uniform(-0.01, 0.01, embed_dim))
    else:
        for i in range(len(wordvocab)):
            vectors.append(np.random.uniform(-0.01, 0.01, embed_dim))
    hit_rate = float(hit)/len(id2word)
    print(("The hit rate is {}".format(hit_rate)))
    return np.array(vectors)