import pandas as pd
import numpy as np
import textacy

#load the entire article text
with open('news_article.txt', 'r') as file:
    data = file.read().replace('\n', '')   
article = data.replace(u'\xa0', u' ')

#create doc object
doc = textacy.make_spacy_doc(article, lang='en_core_web_sm')

#top keyterms from each keyword extraction algorithm
keyterms = ['stock','elect Joe Biden','Trump','Chicago Fed President Charles Evan']



#get document sentences containing keyterms, save to dataframe
#doc.sents gives us a generator object that we can iterate over conatining sentences
#s.lemma_ returns the sentence with all tokens lemmatized
sentences = []
for term in keyterms:
    keyterm_sentences = [s for s in doc.sents if term in s.lemma_]
    for s in keyterm_sentences:
        dictionary = {'keyterm':None,'sentence':None}
        dictionary['keyterm'], dictionary['sentence'] = term, s
        sentences.append(dictionary)
sentence_data = pd.DataFrame(sentences)
sentence_data.to_csv('sentence_data.csv')



#extract relevent SVO patterns using Textacy
#the below line returns a generator, which contains 3-tuples (subject,verb,object) 
SVOs = textacy.extract.subject_verb_object_triples(doc)
svos = []
for term in keyterms:
    for s in SVOs:
        dictionary = {'keyterm':None,'svo':None}
        svo_lemma = [t.lemma_ for t in s]#lemmatize svo patterns
        if term in svo_lemma:
            dictionary['keyterm'],dictionary['svo'] = term, s
            svos.append(dictionary)
        else:
            pass
svo_data = pd.DataFrame(svos)
svo_data.to_csv('svo_data.csv')
print(svo_data)
    
exit()
for term in keyterms:
    keyterm_svos = [s for s in SVOs if term in str(s)]
    print(keyterm_svos)