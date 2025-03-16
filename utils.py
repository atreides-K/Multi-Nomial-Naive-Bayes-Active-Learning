import pickle
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from scipy import sparse as sp


class Vectorizer:
    """
    A vectorizer class that converts text data into a sparse matrix
    """
    def __init__(self,test=False, max_vocab_len=50_000) -> None:
        """
        Initialize the vectorizer
        """
        self.vocab = None
        # TODO: Add more class variables if needed
        self.word2idx =None
        self.idf=None
        self.max_vocab_len=max_vocab_len
        if(test):
            self.test=True

    def fit(self, X_train: np.ndarray) -> None:
        """
        Fit the vectorizer on the training data
        :param X_train: np.ndarray
            The training sentences
        :return: None
        """
        # TODO: count the occurrences of each word
        counts2={}
        for sentence in X_train:
            
            words=sentence.split()
            for word in words:
                counts2[word]=counts2.get(word,0)+1
          
        # TODO: sort the words based on frequency
        sorted_words=sorted(counts2.items(),key=lambda x:(-x[1],x[0]))
        print("total vocab")
        print(len(sorted_words))
        print("max vocab")
        print(self.max_vocab_len)
        
        # TODO: retain the top 10k words
        self.vocab=sorted_words[:self.max_vocab_len]

        #creating a word to inex mapping
        self.word2idx={word: idx for idx, (word, _) in enumerate(self.vocab)}

        # Calculate document frequency for IDF
        num_docs=len(X_train)
        doc_freq={}  # How many documents contain each word
        
        # Count documents that contain each word
        for sentence in X_train:
            unique_words=set(sentence.split())  # Use set to count each word only once per document
            for word in unique_words:
                if word in self.word2idx:
                    doc_freq[word]=doc_freq.get(word, 0) + 1
        
        # Calculate IDF: log(N/df)
        self.idf=np.ones(len(self.word2idx))  # Initialize with ones instead of zeros
        for word,idx in self.word2idx.items():
            df=doc_freq.get(word, 0)
            if df>0:                                      #Avoid log(0)
                # self.idf[idx]=np.log(num_docs/df)
                # Add 1 to numerator and denominator to smooth
                self.idf[idx] = np.log((num_docs + 1) / (df + 1)) + 1
        
            
            
        # raise NotImplementedError

    def transform(self, X: np.ndarray) -> sp.csr_matrix:
        """
        Transform the input sentences into a sparse matrix based on the
        vocabulary obtained after fitting the vectorizer
        ! Do NOT return a dense matrix, as it will be too large to fit in memory
        :param X: np.ndarray
            Input sentences (can be either train, val or test)
        :return: sp.csr_matrix
            The sparse matrix representation of the input sentences
        """
        assert self.vocab is not None, "Vectorizer not fitted yet"
        assert self.idf is not None, "IDF values not calculated yet"
        # if hasattr(self, 'test') and self.test:
        #     # Print the first few sentences of X for debugging
        #     print("First 5 sentences in X:")
        #     for i, sentence in enumerate(X[:5]):
        #         print(f"Sentence {i+1}: {sentence}")
        
        # TODO: convert the input sentences into vectors
        rows,cols,values=[],[],[]

        # TF/CountVec 
        # print("count")
        # for row, sentence in enumerate(X):
        #     words=sentence.split()
        #     word_counts={}
        #     for word in words:
        #         if word in self.word2idx:
        #             word_counts[word]=word_counts.get(word, 0)+1
        #     for word, _count in word_counts.items():
        #         cols.append(self.word2idx[word])
        #         rows.append(row)
        #         values.append(_count)


        
        

        
        
        # TFIDFVEC
        print("tfidf")        
        for row, sentence in enumerate(X):
            words = sentence.split()
            word_counts = {}
            for word in words:
                if word in self.word2idx:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            # Calculate TF (term frequency) for each word in this document
            # and multiply by IDF
            for word, count in word_counts.items():
                idx = self.word2idx[word]
                # TF: count / total words in document
                
                # # raw count
                # tf = count 
                # normalized count
                tf= 1+ np.log(count) if count>0 else 0
                # TF-IDF: TF * IDF
                tfidf = tf * self.idf[idx] 
                
                rows.append(row)
                cols.append(idx)
                values.append(tfidf)
               
                
        
        # Create sparse matrix
        X_sparse = sp.csr_matrix((values, (rows, cols)), shape=(len(X), len(self.word2idx)))

        # loggin output fr understanding
        if hasattr(self, 'test') and self.test:
            print("rows:", rows[:10])
            print("cols:", cols[:10])
            print("values:", values[:10])
        
        if hasattr(self, 'test') and self.test:
            print("sparse matrix shape:", X_sparse.shape)
            print("sparse matrix count:", X_sparse.data[:10])
            print("sparse matrix indices:", X_sparse.indices[:10])
            print("sparse matrix indptr:", X_sparse.indptr[:10])
        return X_sparse
        
        # raise NotImplementedError


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def get_data(
        path: str,
        seed: int
) -> Tuple[np.ndarray, np.ndarray, np.array, np.ndarray]:
    """
    Load twitter sentiment data from csv file and split into train, val and
    test set. Relabel the targets to -1 (for negative) and +1 (for positive).

    :param path: str
        The path to the csv file
    :param seed: int
        The random state for reproducibility
    :return:
        Tuple of numpy arrays - (data, labels) x (train, val) respectively
    """
    # load data
    df = pd.read_csv(path, encoding='utf-8')

    # shuffle data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # split into train, val and test set
    train_size = int(0.8 * len(df))  # ~1M for training, remaining ~250k for val
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    x_train, y_train =\
        train_df['stemmed_content'].values, train_df['target'].values
    x_val, y_val = val_df['stemmed_content'].values, val_df['target'].values
    return x_train, y_train, x_val, y_val
