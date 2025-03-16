import numpy as np
from scipy import sparse as sp


class MultinomialNaiveBayes:
    """
    A Multinomial Naive Bayes model
    """
    def __init__(self, alpha=0.01) -> None:
        """
        Initialize the model
        :param alpha: float
            The Laplace smoothing factor (used to handle 0 probs)
            Hint: add this factor to the numerator and denominator
        """
        # print("alpha",alpha)
        self.alpha = alpha
        self.priors = None
        self.means = None
        self.i = 0  # to keep track of the number of examples seen
        # addition declarations
        self.classes=None

    def fit(self, X: sp.csr_matrix, y: np.ndarray, update=False) -> None:
        """
        Fit the model on the training data
        :param X: sp.csr_matrix
            The training data
        :param y: np.ndarray
            The training labels
        :param update: bool
            Whether to the model is being updated with new data
            or trained from scratch
        :return: None
        """
        
        # np.unique function in NumPy is used to find the unique elements of an array.
        if not update:  # Initialize from scratch
            self.classes= np.unique(y)
            self.no_classes=len(self.classes)
            no_features=X.shape[1]
            
            # Initialize persistent storage for incremental updates
            self.featureFreqInClass=np.zeros((no_features, self.no_classes), dtype=np.float64)
            self.no_featuresInClass= np.zeros(self.no_classes, dtype=np.float64)
            self.total_samples= 0
            
            # These need to be stored for probability calculations
            self.no_features=no_features  # Store feature count for updates
            # self.priorProb=np.zeros(self.no_classes,dtype=np.float64)
            # self.likelihood=np.zeros((self.no_classes,no_features),dtype=np.float64)
            # self.PXc=np.zeros(self.no_classes,dtype=np.float64)??
      
        # Update counts incrementally
        self.total_samples += len(y)
        # debugin stuff
        # List out all column values in X (they are just indices)
        # col_vals = X.indices
        # print("Column values (indices) in X:", col_vals)
        # no_training_sentences=len(y)

        # if hasattr(self,'test') and self.test:
        # print(X.shape)
        # print(X.shape[1])


        

        # featureFreqInClass=np.zeros((no_features,self.no_classes),dtype=np.float64) #n
        # no_featuresInClass=np.zeros(self.no_classes,dtype=np.float64) #N
        
        
        # for i,class_ in enumerate(self.classes):
        #     # y = np.array([1, 2, 3, 2, 1, 2, 3])
        #     # class_ = 2
        #     # rows = np.where(y == class_)[0]
        #     # print(rows)

        #     rows=np.where(y==class_)[0]
        #     self.priorProb[i]=len(rows)/no_training_sentences
        #     # for row in rows:
        #     #     f,Feature=self.FeatData(X,row)

        #     #     for k in range(len(f)):
        #     #         # Ncf
        #     #         featureFreqInClass[f[k],class_]+=Feature[k]
        #     #         # Nc
        #     #         no_featuresInClass[class_]+=Feature[k]
            
        #     # Extractinh all features for this class at once (more efficient)
        #     class_features = X[rows]
        #     featureFreqInClass[:, class_] = np.array(class_features.sum(axis=0)).flatten()
        #     no_featuresInClass[class_] = class_features.sum()
        
        
        
        
        # optimized code fr above ty shivam
        batch_featureFreq=np.zeros((self.no_features, self.no_classes),dtype=np.float64)
        batch_featuresInClass=np.zeros(self.no_classes,dtype=np.float64)
        
        for i,class_ in enumerate(self.classes):
            rows=np.where(y==class_)[0]
            class_features=X[rows]
            # to accumate all features here
            batch_featureFreq[:, i]=np.array(class_features.sum(axis=0)).flatten()
            batch_featuresInClass[i]=class_features.sum()
        
        self.featureFreqInClass+=batch_featureFreq
        self.no_featuresInClass+=batch_featuresInClass



        # for class_ in range(0,self.no_classes):
        #     # for feature in range(no_features):
        #     #     self.likelihood[class_, feature]=(featureFreqInClass[feature,class_]+self.alpha)/(no_featuresInClass[class_]+self.alpha*no_features)
        #     self.likelihood[class_] = (featureFreqInClass[:, class_] + self.alpha)/(no_featuresInClass[class_] + self.alpha * no_features)

        # optimizin calc
        self.priorProb=(np.array([np.sum(y == cls) for cls in self.classes])+self.alpha)/(self.total_samples+self.alpha*self.no_classes)
    
        # Calculate likelihood with Laplace smoothing
        self.likelihood = (self.featureFreqInClass.T + self.alpha)/(self.no_featuresInClass[:,None]+self.alpha* self.no_features)
        # Debugging information to understand why tf-idf is giving lesser accuracy
        # print("Feature frequencies in each class:")
        # for class_ in (self.classes):
        #     print(f"Class {class_}:")
        #     for feature in range(no_features):
        #         print(f"Feature {feature}: {featureFreqInClass[feature, class_]}")
        # print("Number of features in each class:")
        # for class_ in (self.classes):
        #     print(f"Class {class_}: {no_featuresInClass[class_]}")


        

            
        
    
    def FeatData(self,X,obs_prob):
        j=list(range(X.indptr[obs_prob],X.indptr[obs_prob+1]))
        f=X.indices[j]
        Features=X.data[j]
        # if Features.size == 0:
        #     maxF = 1  # avoid division by zero
        # else:
        #     maxF = np.max(Features)

        return (f,Features)

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        """
        Predict the labels for the input data
        :param X: sp.csr_matrix
            The input data
        :return: np.ndarray
            The predicted labels
        """
        assert self.priorProb.shape[0] == self.likelihood.shape[0]
        preds = []
        for i in range(X.shape[0]):
            class_scores=np.zeros(self.no_classes)
            
            # Get features for this data point
            f,features=self.FeatData(X, i)
            
            # Calculate log probabilities for each class
            for c in range(self.no_classes):
                # Start with prior probability (in log space)
                class_scores[c]=np.log(self.priorProb[c])
                
                # Add log likelihood for each feature present in this data point
                for j in range(len(f)):
                    feature_idx= f[j]
                    # # 1. Multiply by feature value
                    # feature_val = features[j]
                    # # Multiply by feature value (which is equiv to adding logs)
                    # class_scores[c]+=feature_val * np.log(self.likelihood[c, feature_idx])

                    # 2. Use binary presence
                    class_scores[c]+=np.log(self.likelihood[c, feature_idx])
            
            # Choose the class with highest probability
            predicted_class = self.classes[np.argmax(class_scores)]
            preds.append(predicted_class)
        
        return np.array(preds)
    
    def predict_prob(self, X: sp.csr_matrix) -> np.ndarray:
        # Precompute during fit()
        log_likelihood = np.log(self.likelihood)  # (classes x features)
        log_prior = np.log(self.priorProb)        # (classes,)
        
        # Vectorized computation
        log_probs = X.dot(log_likelihood.T) + log_prior  # (samples x classes)
        
        # Stable softmax
        max_log = np.max(log_probs, axis=1, keepdims=True)
        exp_log = np.exp(log_probs - max_log)
        return exp_log / exp_log.sum(axis=1, keepdims=True)

