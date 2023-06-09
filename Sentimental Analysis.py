import nltk
import movie_reviews
import csv
import pandas as pd
import dictionary as dict

with open('IMDB_dataset_320.000_reviews.csv', mode="r") as csv_file: 
    reader = csv.reader(csv_file) 

    for item in reader:
        print(item)

movie_reviews.words()
len(movie_reviews.words())

nltk.FreqDist(movie_reviews.words())
nltk.FreqDist(movie_reviews.words())['happy']
nltk.FreqDist(movie_reviews.words()).most_common(15)
movie_reviews.fileids()
movie_reviews.fileids('pos')
movie_reviews.fileids('neg')

movie_reviews.words('neg/textword_1.txt')

all_words = nltk.FreqDist(movie_reviews.words())
feature_vector = list(all_words)[:5000]
def find_feature(word_list):
        feature = {}
review = movie_reviews.words('neg/textword_2.txt')
for x in range(len(feature_vector)):
    [feature_vector[x]] = feature_vector[x] in review
    list(feature_vector)
    if feature[x] == True:
          document = [(movie_reviews.words(file_id),category) 
               for file_id in movie_reviews.fileids() 
               for category in movie_reviews.categories(file_id)]

find_feature(document[0][0])
feature_sets = [(find_feature(word_list),category) 
                 for (word_list,category) in document]
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
from sklearn import model_selection
train_set,test_set = model_selection.train_test_split(feature_sets,test_size = 0.25)
print(len(train_set))
print(len(test_set))
model = SklearnClassifier(SVC(kernel = 'linear'))
model.train(train_set)
accuracy = nltk.classify.accuracy(model, test_set)
print(f'VC Accuracy : {accuracy}')
