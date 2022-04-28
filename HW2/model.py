import math
from collections import Counter, defaultdict
from typing import List

import nltk
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm



class Ngram:
    def __init__(self, config, n=2):
        self.tokenizer = ToktokTokenizer()
        self.n = n
        self.model = None
        self.config = config

    def tokenize(self, sentence):
        '''
        E.g.,
            sentence: 'Here dog.'
            tokenized sentence: ['Here', 'dog', '.']
        '''
        return self.tokenizer.tokenize(sentence)

    def get_ngram(self, corpus_tokenize: List[List[str]]):
        '''
        Compute the co-occurrence of each pair.
        '''
        # begin your code (Part 1)
        
        return_model = {}
        
        return_num = {}
        
        features = []
        
        for sentences in corpus_tokenize:
            # sentences[0] is [cls] doesn't need to count
            for i in range(1, len(sentences) - 1):
                features += [(sentences[i],sentences[i+1])]
                
                #if first has been in dict
                if return_model.__contains__(sentences[i]):

                    return_num[sentences[i]] +=1

                    #if second hasn't been in dict[first]
                    
                    if return_model[sentences[i]].__contains__(sentences[i+1]):
                        return_model[ sentences[i] ][ sentences[i+1] ] += 1
                    else:
                        return_model[ sentences[i] ][ sentences[i+1] ] = 1
                
                else: 
                    #new a dict 
                    return_num[sentences[i]] = 1
                    #make it a 2d dict
                    return_model[sentences[i]] = {}
                    return_model[sentences[i]][sentences[i+1]] = 1
                    
                    
        for sentences in return_model :
            num = return_num[sentences]
            
            for sentence in return_model[sentences] :
                return_model[sentences][sentence] /= num
           
    
    
        return return_model, features
       
        # end your code
  
    def train(self, df):
        '''
        Train n-gram model.
        '''
        corpus = [['[CLS]'] + self.tokenize(document) for document in df['review']]     # [CLS] represents start of sequence
        
        # You may need to change the outputs, but you need to keep self.model at least.
        self.model, self.features = self.get_ngram(corpus)
        

    def compute_perplexity(self, df_test) -> float:
        '''
        Compute the perplexity of n-gram model.
        Perplexity = 2^(entropy)
        '''
        if self.model is None:
            raise NotImplementedError("Train your model first")

        corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        
        # begin your code (Part 2)

        logsum = 0
        different = 0
        
        for sentence in corpus:            
            different += len(sentence)
            for i in range(len(sentence) - 1):
                
                if self.model.__contains__(sentence[i]):
                    
                    if self.model[ sentence[i] ].__contains__(sentence[i+1]):
                        pr = self.model[ sentence[i] ][ sentence[i+1] ]
                    else:
                        pr = 1 / len(self.model[sentence[i]])

                    
                    logsum += (math.log(pr, 2))
                        
        perplexity = 2**( - ( logsum / different ) )
        
        
        # end your code

        return perplexity

    def train_sentiment(self, df_train, df_test):
        '''
        Use the most n patterns as features for training Naive Bayes.
        It is optional to follow the hint we provided, but need to name as the same.

        Parameters:
            train_corpus_embedding: array-like of shape (n_samples_train, n_features)
            test_corpus_embedding: array-like of shape (n_samples_train, n_features)
        
        E.g.,
            Assume the features are [(I saw), (saw a), (an apple)],
            the embedding of the tokenized sentence ['[CLS]', 'I', 'saw', 'a', 'saw', 'saw', 'a', 'saw', '.'] will be
            [1, 2, 0]
            since the bi-gram of the sentence contains
            [([CLS] I), (I saw), (saw a), (a saw), (saw saw), (saw a), (a saw), (saw .)]
            The number of (I saw) is 1, the number of (saw a) is 2, and the number of (an apple) is 0.
        '''
        # begin your code (Part 3)

        # step 1. select the most feature_num patterns as features, you can adjust feature_num for better score!
        
        feature_num = 500
        
        #calcu all features and its appearance times
        features = Counter(self.features)
        
        features = sorted(features.items(),key = lambda item:item[1], reverse=True)
        
        feature_num = min(feature_num, len(features))
        features = features[0:feature_num]

        # step 2. convert each sentence in both training data and testing data to embedding.
        # Note that you should name "train_corpus_embedding" and "test_corpus_embedding" for feeding the model.
        
        train_corpus_embedding = []
        test_corpus_embedding = []
        
        train_corpus = [['[CLS]'] + self.tokenize(document) for document in df_train['review']] 
        test_corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']] 


        
        for sentences in train_corpus:
            
            embedded_arr = []
            bi_train_arr = []
            #list of tuple of first word and second
            for i in range(len(sentences) - 1):

                bi_train_arr += [(sentences[i], sentences[i+1])]
            
            for feature in features:
                key = feature[0]
                if key in bi_train_arr:
                    num = bi_train_arr.count(key)
                    embedded_arr += [num]
                else:
                    embedded_arr += [0]

            train_corpus_embedding.append(embedded_arr)
            
        train_corpus_embedding = np.array(train_corpus_embedding)
        
        
        
        for sentences in test_corpus:
            embedded_arr = []
            bi_test_arr = []
            #list of tuple of first word and second
            for i in range(len(sentences) - 1):

                bi_test_arr += [(sentences[i], sentences[i+1])]
            
            for feature in features:
                key = feature[0]
                if key in bi_test_arr:
                    num = bi_test_arr.count(key)
                    embedded_arr += [num]
                else:
                    embedded_arr += [0]
            
            test_corpus_embedding.append(embedded_arr)
            
        test_corpus_embedding = np.array(test_corpus_embedding)
        
        
        df_train['sentiment'] = np.array(df_train['sentiment'])
        df_test['sentiment'] = np.array(df_test['sentiment'])
        
        # end your code
        
        

        # feed converted embeddings to Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(train_corpus_embedding, df_train['sentiment'])
        y_predicted = nb_model.predict(test_corpus_embedding)
        precision, recall, f1, support = precision_recall_fscore_support(df_test['sentiment'], y_predicted, average='macro', zero_division=1)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")


if __name__ == '__main__':
    '''
    Here is TA's answer of part 1 for reference only.
    {'a': 0.5, 'saw: 0.25, '.': 0.25}

    Explanation:
    (saw -> a): 2
    (saw -> saw): 1
    (saw -> .): 1
    So the probability of the following word of 'saw' should be 1 normalized by 2+1+1.

    P(I | [CLS]) = 1
    P(saw | I) = 1; count(saw | I) / count(I)
    P(a | saw) = 0.5
    P(saw | a) = 1.0
    P(saw | saw) = 0.25
    P(. | saw) = 0.25
    '''

    # unit test
    test_sentence = {'review': ['I saw a saw saw a saw.']}
    model = Ngram(2)
    model.train(test_sentence)
    print(model.model['saw'])
    print("Perplexity: {}".format(model.compute_perplexity(test_sentence)))
    
