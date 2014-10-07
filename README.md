mldemos
=======

Machine Learning Demos

## Vectorizer

This demo is from the book *Building Machine Learning Systems with Python*.

A small training set resides in `data/`, which simulates thread titles on a forum. Hard coded in the script is a new thread title, which the script uses to find the most matching thread from `data/`. The use case is finding similar threads in a forum, search results on a website, etc.

Dependencies:

* scipy
* numpy
* scikit-learn
* ntlk

First each title is first *stemmed* by a [Stemmer](http://en.wikipedia.org/wiki/Stemming). A stemmer will turn each inflected or derived word into it's *stem word*, i.e. `imaging`, `image` and `images` are all turned into the stem word `imag`. This is so the algorithm can compare threads which uses different words to describe the same thing.

When all words have been stemmed, a [TF-IDF](http://en.wikipedia.org/wiki/Tf%E2%80%93idf) Vectorizer is used to calculate the importance of each word in the corpus. In the example posts in this demo it doesn't work too well since the corpuses are very short. What it does is weigh each word based on how often and how frequently it appears in the corpus, to filter out *common* words.

After stemming and calculating TF-IDF, each post is put into a vector with its word stem and TF-IDF value. Then the new post is *transformed* using the vectorized who's been *trained* with the test data. If no words match any of the test data, the vectorizer will output an empty vector. If it contains related words a vector with the new post's word stems and TF-IDFs is outputed in a new vector, which is matched against all the vectors of the tests posts. The test post with closes normalized distance (`dist_norm()`) will be most similar to the new post.
