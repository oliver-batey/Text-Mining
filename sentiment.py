import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

def sentiment_polarity(string):
    polarity =  TextBlob(string).sentiment[0]
    return polarity
    

#load svo patterns and sentences
svo_data = pd.read_csv('svo_data.csv',index_col=0)
sen_data = pd.read_csv('sentence_data.csv',index_col=0)

#apply sentiment(score) to the sentence column
sen_data['sentiment_polarity'] = sen_data.sentence.apply(sentiment_polarity)

#print mean seentiment polarity score
mean_sentiment_scores = sen_data.groupby('keyterm').mean()

#plot results
labels = [l.replace(' ','\n') for l in mean_sentiment_scores.index.to_list()]
pos = np.arange(len(labels))
ax = mean_sentiment_scores.plot(kind='bar',
                    color='lightblue',
                    edgecolor='midnightblue',
                    legend=False,
                    rot=0)
plt.axhline(0,0,1,linestyle='--',color='grey',lw=0.5)
ax.set_xticks(pos)
ax.set_xticklabels(labels)
plt.ylabel('Mean Sentiment Polarity')
plt.tight_layout()
plt.savefig('mean_keyterm_sentiment.png',dpi=300)
plt.show()

exit()
