import nltk.stem
import os
import sys
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

DIR='posts/'

def dist_norm(v1, v2):
    v1_n = v1/sp.linalg.norm(v1.toarray())
    v2_n = v2/sp.linalg.norm(v2.toarray())
    delta = v1_n - v2_n
    return sp.linalg.norm(delta.toarray())

"""
# if a word occures multiple times it will be regarded as being closer
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: (stemmer.stem(w) for w in analyzer(doc))
"""

# TF-IDF Vectorizer doesn't regard multiple counts of words as closer
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (stemmer.stem(w) for w in analyzer(doc))

posts = [open(os.path.join(DIR, f)).read() for f in os.listdir(DIR)]
stemmer = nltk.stem.SnowballStemmer('english')
vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english', decode_error='ignore')
X_train = vectorizer.fit_transform(posts)

new_post = 'imaging databases'
new_post_vec = vectorizer.transform([new_post])
best_dist = sys.maxint
best_i = None

for i in range(0, len(posts)):
    post = posts[i]
    if post == new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist_norm(post_vec, new_post_vec)
    print("=== Post %i with dist=%.2f: %s" % (i, d, post))
    if d < best_dist:
        best_dist = d
        best_i = i

print("Best post is %i with dist=%.2f" % (best_i, best_dist))











