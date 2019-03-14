import nltk
import collections
import numpy
from nltk.stem  import LancasterStemmer
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import ngrams,ne_chunk,wordpunct_tokenize,pos_tag,FreqDist

with open('data_set.txt', 'r') as f:
  raw=f.read()

#Tokenization of the text
word_tokens=nltk.word_tokenize(raw)
print(word_tokens)

#lemmatization of the text
lemmatizer = WordNetLemmatizer()
print("Lemmatization ------------------------------------------------------------:\n")

for tok in word_tokens:
  print(lemmatizer.lemmatize(str(tok)))

#printing trigrams
print("Trigrams -------------------------------------------------------------------:\n")
trigram = []
x=0
trigram.append(list(ngrams(word_tokens, 3)))
print(trigram)

TrigramsOutput = []
for big in ngrams(word_tokens, 3):
    # getting the trigrams using ngrams and iterating them
    TrigramsOutput.append(big)
print(TrigramsOutput)

wordFreq = FreqDist(TrigramsOutput)
print(wordFreq)
most_Common_word = wordFreq.most_common()
print(most_Common_word)

#top 10 most repeated trigrams
most_Common10_words = wordFreq.most_common(10)
print(most_Common10_words)

#print sentences
sent_Tokens = sent_tokenize(raw)
print(sent_Tokens)

#array to append the sentences
concatenatedArray = []
#iterating through sentences
for sentence in sent_Tokens:
    #going through the present trigrams
    for a, b ,c in TrigramsOutput:
        #top 5 trigrams
        for ((c, m, n ), length) in most_Common10_words:
            #comparing them with top 5 trigrams
            if(a, b, c == c, m, n):
                #appending the array
                concatenatedArray.append(sentence)
#printing the concatenated result
print("Max of Concatenated Array : ", max(concatenatedArray))