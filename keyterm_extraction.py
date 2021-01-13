import pandas as pd
import numpy as np
import textacy
from textacy import ke
import matplotlib.pyplot as plt




def keyterm_dataframe(keyterm_list,algorithm_name):
    
    dict_list = []
    for kt in keyterm_list:
        dictionary = {'keyterm':None,'score':None}
        dictionary['keyterm'] = kt[0]
        dictionary['score'] = kt[1]
        dict_list.append(dictionary)
        df = pd.DataFrame(dict_list)
        df['algorithm']=algorithm_name
    return df


def decompose_keyterms(keyterm_list):
    terms = [el[0].replace(' ', '\n') for el in keyterm_list]
    scores = np.asarray([el[1] for el in keyterm_list])
    return terms, scores


def make_barplot(scores,keyterms,ax=None,
                    title='barplot',
                    ylabel='ylabel',
                    color='lightblue',
                    edgecolor='midnightblue',
                    align='center',
                    alpha=1.0):
                    
    bars = ax.bar(np.arange(len(keyterms)),
            scores,
            align=align,
            color=color,
            alpha=alpha)
    for bar in bars:
        bar.set_edgecolor(edgecolor)

    ax.set_xticks(np.arange(len(keyterms)))
    ax.set_xticklabels(keyterms,fontsize=5)
    ax.set_ylabel(ylabel,fontsize=12)
    ax.set_title(title,fontsize=12)
    return ax
    



#open data from .txt file
with open('news_article.txt', 'r') as file:
    data = file.read().replace('\n', '')   
article = data.replace(u'\xa0', u' ')

#create doc object
doc = textacy.make_spacy_doc(article, lang='en_core_web_sm')

#KEYTERM EXTRACTION
#Each algorithm returns a list of tuples, containg the keyterm and a score
textrank = ke.textrank(doc,normalize="lemma")
yake = ke.yake(doc,normalize="lemma")
scake = ke.scake(doc,normalize="lemma")
sgrank = ke.sgrank(doc,normalize="lemma")

#separate terms and relevany scores
terms_textrank, scores_textrank  = decompose_keyterms(textrank)
terms_yake, scores_yake  = decompose_keyterms(yake)
terms_scake, scores_scake  = decompose_keyterms(scake)
terms_sgrank, scores_sgrank  = decompose_keyterms(sgrank)

#save results to dataframe
df = keyterm_dataframe(scake,'scake')
print(df)
    


#Make plot
# fig, axes = plt.subplots(2,2,figsize=(11,8))
# make_barplot(scores_textrank, terms_textrank,axes[0,0],title='TextRank algorithm',ylabel='Importance')
# make_barplot(scores_yake,terms_yake,axes[0,1],title='YAKE algorithm',ylabel='Importance',color='lightcoral',edgecolor='firebrick')
# make_barplot(scores_scake,terms_scake,axes[1,0],title='sCAKE algorithm', ylabel='Importance',color='springgreen',edgecolor='darkgreen')
# make_barplot(scores_sgrank,terms_sgrank,axes[1,1],title='SGRank algorithm', ylabel='Importance',color='moccasin',edgecolor='darkorange')
# plt.tight_layout()
# plt.savefig('keyword_plots.png',dpi=300)
# plt.show()



svo = textacy.extract.subject_verb_object_triples(doc)
for i in svo:
    print(i)


print('*'*20)

#find the keywords
stock_context = textacy.text_utils.keyword_in_context(article,'stock')
biden_context = textacy.text_utils.keyword_in_context(article,'Biden')
for j in stock_context:
    print(j)

print('*'*20)
  
for j in biden_context:
    print(j)