from sklearn.datasets import load_files
from sklearn.ensemble import RandomForestClassifier
# import pdb; pdb.set_trace()
keyboard = load_files('./data')
key_data, key_target = keyboard.data, keyboard.target

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(key_data)
print("*** X_train_counts Shape\n", x_train_counts.shape)

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(x_train_counts)
x_train_tf = tf_transformer.transform(x_train_counts)
print("*** X_train_tf shape", x_train_tf.shape)

tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
print("*** X_train_tfidf", x_train_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(x_train_tfidf, key_target)

docs_new = [
	'y y o u o u space space alt tab tab tab alt tab m m a y a y space space h h a a v e v e t t r r a a i i n n e e d d space space m m e e backspace backspace o o d d e e l l s s space space u u s s i i n n g g space space k k - - f f o l o l s d s d space space c c r r o o s s s s space space v v a a l l i i d d a a t t i i o n o n space space',
	' c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c '	
]

x_new_counts = count_vect.transform(docs_new)
x_new_tfidf = tf_transformer.transform(x_new_counts)
predicted = clf.predict(x_new_tfidf)
for docs, cat in zip(docs_new, predicted):
	print('%r => %s' % (docs, keyboard.target_names[cat]))

import joblib
joblib.dump(clf, 'catModel.pkl', compress=9)
joblib.dump(count_vect, 'CountVectorizer')
joblib.dump(tf_transformer, 'tf_transformer')

model_clone = joblib.load('catModel.pkl')
vect_clone = joblib.load('CountVectorizer')
transformer_clone = joblib.load('tf_transformer')

clone_counts = vect_clone.transform(docs_new)
clone_tfidf = transformer_clone.transform(clone_counts)
clone_predict = model_clone.predict(clone_tfidf)
for docs, cat in zip(docs_new, clone_predict):
	print('-----%r => %s' % (docs, keyboard.target_names[cat]))