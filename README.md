# SentimentAnalysis_Twitter

INTRODUCTION
Humans love to express their opinions and sentiments. This opinion can be communicated through different forms like speech, body language, facial expressions, and text. Understanding users’ view or opinion by extracting subjective texts is known as Sentiment Analysis. While applications of sentiment analysis started out as curiosity-driven exercises, it has moved on to tasks which provide real value to organizations. One of the main tasks of Sentiment Analysis is to classify the polarity of a text, which represents whether the opinion expressed toward the subject in the piece is positive, negative, or neutral. Social networking sites like Twitter leverage Sentiment classification which acts as a rich source of information. In this report, I’ll share techniques and features I’ve used to build multiple classifiers to predict the overall sentiment of a tweet as positive, negative, and neutral and their corresponding evaluation results.	

DATASET
Dataset has been taken from Semeval 2017 (https://alt.qcri.org/semeval2017/task4/), an online competition, to build the sentiment classifiers. Dataset has been divided into 3 categories: Training, development and testing set. 
Dataset Type	Dataset Size
Training	    45101 tweets
Development 	2000 tweets
Testset1	    3531 tweets
Testset2	    1853 tweets
Testset3	    2379 tweets
Each dataset has 3 columns, tweet-id, sentiment, and tweet text. Training dataset is unbalanced as neutral tweets are more than positive and negative tweets. There are no null values/missing samples in the datasets.     

PRE-PROCESSING
To pre-process tweets, I’ve implemented below techniques for tweets:
•	Removed contractions e.g. “It’s” to “It is” and “wont” to “will not”.
•	Lowercase characters
•	Substituting for RT at the start of retweeted tweets with space.
•	Removing URLs, digits, and punctuations.
•	Substituting hashtag symbol (#) with space.
•	Substituting multiple spaces with one space.
•	Removing @ mentions with space.
•	Substituting new lines with space.
•	De-emojify the text i.e., removing any emojis like smileys with space.
•	Stemming tokens. 
•	Removing one length words.
While pre-processing, I’ve used stemming rather than lemmatization as the accuracy of the model improved with stemming over lemmatization by around 2%. Also, I’ve not removed stop words as sentences like ‘I am not happy’ and ‘I am happy’ will be tagged under same sentiment if stop word ‘not’ is removed. Stop words like not, never, under etc. represent important information for our use case. 
FEATURE EXTRACTION
As raw text data can’t be fed directly into the machine learning algorithms, we need to convert text data into numerical features. To do so, we run feature extraction algorithms. Below are the feature extraction techniques which are required before training the classifiers. 
•	Bag-of-words (BOW): The bag-of-words representation turns text into fixed-length vectors which have the count of how many times each word appears. There are 2 steps involved in this, creating a vocabulary, and determining the count. BOW representation doesn’t contain the context, it only determines the number of words in the document. To implement BOW, I’ve used scikit-learn’s CountVectorizer class.  
•	TF-IDF: TF-IDF refers to term frequency -inverse document frequency. It denotes how relevant a word is to a document. Term frequency calculates how many times word appears in the document and inverse document frequency calculates the frequency of word across a set of documents. To find the important words which might be less frequently used in the set of documents, we use TF-IDF. To implement TF-IDF, I’ve used scikit-learn’s TfidfVectorizer class.
•	GloVe: Global Vectors for Word Representation. An unsupervised learning algorithm that uses vector representations to find the semantic similarity between the words. I’ve used GloVe to obtain the pre-trained word vectors 6B tokens, 100D from (https://nlp.stanford.edu/projects/glove/)

CLASSIFIERS
To develop the classifiers, I’ve used three traditional machine learning models (Naïve Bayes, Random Forest, Logistic Regression) and one neural network model with bi-directional LSTM layer. 
USING TRADITIONAL MACHINE LEARNING CLASSIFIERS
After pre-processing the tweet column of dataset for training, development, and test datasets, feature extraction algorithms are run. For training traditional machine learning classifiers, feature generation has been achieved by either using tf-idf or bag-of-words.
	Feature generation using TF-IDF: Tf-idf feature generation is implemented using TfidfVectorizer() class which generates vector representation for unigram, bigram and trigram features. Max-features has been set to 5000 which means it will consider the top max_features ordered by term frequency across the corpus. After this, the training set tweets are fit on TfidfVectorizer() object which generates vocabulary. On the vocabulary, training, development and test sets are transformed to output fixed vector representation of size (n_samples, max_features) with weights assigned to each feature. get_feature_names_out() outputs a list in which the ngrams appear according to the column position of each feature. 
	Eeature generation using Bag-of-words (BOW): For BOW, CountVectorizer() class is used which is initialized with parameters for max_features=5000, n_gram=(1,3), min_df =1 and max_df=0.9. This will generate count of occurrences of tokens as a fixed vector representation of (n_samples, max_features) size. 

1.	Random Forest classifier:  Splitting the dataset hierarchically based on a condition of the attribute value is how decision tree works. Random Forest fits data on large number of decision tree classifiers. At each node, data is split based on whether the term exists or not. The tree is traversed in top-down manner. Averaging all the predictions from the trees gives the final prediction for the class label. 

In the classifier I’ve built, I’ve used scikit-learn’s RandomForestClassifier with n_estimators=100 which are the number of trees on which the data would be trained. After generating fixed vector representation from either TF-IDF or BOW features, random forest classifier is trained. 

2.	Naïve Bayes classifier: Naïve Bayes predicts the probability of a class with assumption that all features all independent. It is a supervised learning algorithm based on Bayes’ theorem. As we need to predict multiple classes, scikit-learn’s MultinomialNB() class is used for training.

3.	Logistic Regression: Logistic Regression is a supervised machine learning algorithm used to predict the likelihood of an event occurring. It uses logistic function to model the output class. The input variables must be independent of each other. As we need to perform multi-class classification on sentiments, LogisticRegression() class is used which accepts features and class labels of the training set. After generating fixed vector representation from either TF-IDF or BOW features, model is trained. 

HYPERPARAMETER TUNING
•	While feature extraction, accuracy improved significantly by using (ngram_range) = (1,3) instead of (1,1) and (1,2). This shows that classifier performs better with trigrams included as features. 
•	max_df parameter eliminates words which have large number of occurrences. max_df after tuning has been set to 0.9
•	max_features parameter while feature extraction sets the vocabulary that only consider the top max_features ordered by term frequency. max_features = 5000 has been set to consider only top 5000 words and also in interest of time, the max_features have been limited. 
•	n_estimators for random forest classifier sets the number of trees. The number of trees after tuning have been set to 100. If we increase the n_estimators by a large value, overfitting of data can happen. 

NEURAL NETWORK WITH LSTM LAYER
Neural networks provide a modern and successful approach to text classification problems. The basic unit in NN is a neuron or node. The input is a vector representation of the features. The connections between nodes have associated weights(w). Weights can be either positive or negative depending on the function they need to perform, i.e., excite or inhibit the connection. When a neural net is trained, weights are set to a random value and are continually updated until the loss is minimized. 
As with traditional ML approaches, first the pre-processing of the tweet column of the training, development, and test datasets is done. After pre-processing, tokens are generated using Tokenizer from Keras. Tokenizer will generate a vocabulary of terms and will sequence and pad the tweets. To extract features for texts, GloVe word embedding is being used. GloVe has already generated word vectors for 6B tokens which can be utilized for our use case of sentiment classification. Embedding matrix containing word and features is generated which is of the size (max_features, embedding_size). We have initialized max_features to 5000 and embedding_size is 100. 
To implement Neural network, Pytorch’s nn.Module class has been leveraged. There are 6 layers which have been added to the neural network. 
1.	Embedding layer: Embedding layer takes (max_features, embedding_size) as input. Weights are initialized and assigned to all the input connections.
2.	LSTM layer: Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence. A Bidirectional LSTM is a sequence processing model that consists of two LSTMs: one taking the input in a forward direction, and the other in a backwards direction. BiLSTMs effectively increase the amount of information available to the network, improving the context available to the algorithm.  
3.	Linear layer: Linear layer does the average pooling and then max pooling of the outputs from LSTM layer. Max pooling is a pooling operation that selects the maximum element from the region of the feature map covered by the filter.   
4.	ReLU layer: Non-linear layer which performs activation using Rectified Linear Unit (ReLU) function.  It will output the input directly if it is positive, otherwise, it will output zero.
5.	Dropout layer: Dropout layer is used to avoid overfitting of training data.  It is an effective technique for regularization.
6.	Output layer: Linear Layer with output as matrix of (hidden_layer_size, n_classes). 

HYPERPARAMETER TUNING
•	Learning rate helps set the amount that the weights are updated during training. When learning rate was set to 0.1, no change in loss was seen and hence, F1 score was inferred as 0. Changing learning rate to 0.003 significantly improved the F1 score as the loss decreased and val_acc increased. This shows how important learning rate hyperparameter is.
•	The Dropout layer applies Dropout to the input. It randomly sets input units to 0 with a frequency rate given at each step during training time, which helps prevent overfitting. Dropout layer rate =0.5 is set after tuning. The F1 score showed improvement when setting dropout layer rate from 0.1 to 0.5. 
•	After multiple executions, n_epochs = 6 generated the highest accuracy without overfitting the training data. The validation loss increased after increasing the n_epochs more than 6. 
PREDICTIONS
After training the classifiers and tuning the parameters, the next step is to run predictions for unseen data. We’ve been provided with three testsets. 
For traditional machine learning models, we first do feature extraction for testsets using tf-idf or bag-of-words and then run classifier.predict() function on feature_vector_test. This will generate class labels which are further evaluated to test the accuracy of the model.
For neural network, the preprocessed testsets are tokenized and and padded using Keras. The tweets are further converted to tensors on which NN model is run to get predictions. Model(x).detach() is used to run predictions. Softmax() function is run on the predictions to normalize the values. The prediction with the highest value is chosen as the final predicted class label.

EVALUATION AND RESULTS
To see how accurate the model predictions are, we are running evaluation script which produces the macroaveraged F1 score on the testsets. 
Comparing performance of classifiers on 3 testsets:
Naïve Bayes	Features used	Twitter-test1	Twitter-test2	Twitter-test3
	            TF-IDF	     0.415	          0.448	        0.401
	            Bag-of-words 0.534	          0.581	        0.519

Random Forest	Features used	Twitter-test1	Twitter-test2	Twitter-test3
	              TF-IDF	      0.422	        0.444	        0.385
	              Bag-of-words	0.427	        0.471       	0.372

Logistic Regression	Features used	Twitter-test1	Twitter-test2	Twitter-test3
	              TF-IDF	       0.522	       0.539	      0.487
	              Bag-of-words	 0.546	         0.574	      0.499

NN with bi-LSTM	Features used	Twitter-test1	Twitter-test2	Twitter-test3
	               GloVe	        0.551	        0.547	      0.549

The maximum F1-Score (0.581) was achieved by Naïve Bayes using Bag-of-words feature on Twitter-test2 whereas Random Forest with TF-IDF features performed the worst on Twitter-test3 with score of 0.372.
Overall, NN with bi-LSTM layer performed the best with average F1-score of 0.549 calculated over the 3 testsets followed by Logistic Regression with Bag-of-words features with average score of 0.539.  
Comparing classifiers performance on TF-IDF features using average F1-Score on 3 testsets:
Classifier	Features used	Average F1-Score of 3 testsets
Naïve Bayes	TF-IDF	0.421
Random Forest	TF-IDF	0.417
Logistic Regression	TF-IDF	0.516

Logistic Regression performed the best on TF-IDF features.
Comparing classifiers performance on Bag-of-words features using average F1-Score on 3 testsets:
Classifier	Features used	Average F1-Score of 3 testsets
Naïve Bayes	Bag-of-words	0.544
Random Forest	Bag-of-words	0.423
Logistic Regression	Bag-of-words	0.539

Naïve Bayes performed the best with Bag-of-words features.

