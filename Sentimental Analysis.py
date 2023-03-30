import nltk
from nltk.corpus import movie_reviews

movie_reviews.words()
len(movie_reviews.words())

nltk.FreqDist(movie_reviews.words())
nltk.FreqDist(movie_reviews.words())['happy']
nltk.FreqDist(movie_reviews.words()).most_common(15)
movie_reviews.fileids()
movie_reviews.fileids(‘pos’)
movie_reviews.fileids(‘neg’)

movie_reviews.words(‘neg/cv001_19502.txt’)

all_words = nltk.FreqDist(movie_reviews.words())
feature_vector = list(all_words)[:4000]
feature = {}
review = movie_reviews.words(‘neg/cv954_19932.txt’)
for x in range(len(feature_vector)):
 feature[feature_vector[x]] = feature_vector[x] in review
 [x for x in feature_vector if feature[x] == True]
 document = [(movie_reviews.words(file_id),category) for file_id in movie_reviews.fileids() for category in movie_reviews.categories(file_id)]document
 
 find_feature(document[0][0])
 feature_sets = [(find_feature(word_list),category) for (word_list,category) in document]
 from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
from sklearn import model_selection
train_set,test_set = model_selection.train_test_split(feature_sets,test_size = 0.25)
print(len(train_set))
print(len(test_set))
model = SklearnClassifier(SVC(kernel = ‘linear’))
model.train(train_set)
accuracy = nltk.classify.accuracy(model, test_set)print(‘SVC Accuracy : {}’.format(accuracy))
