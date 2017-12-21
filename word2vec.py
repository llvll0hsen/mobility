import json
import os

import pandas as pd
import cPickle
import numpy as np
from PreprocessTags import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim, logging


from util import output_path_files, output_path_plots
def clean_noise(tags):
    if 'establishment' in tags:
        tags = tags.replace('establishment','')
    if 'point_of_interest' in tags:
        tags = tags.replace('point_of_interest','')
    return tags.strip()

def format_tags(tags):
    tags_string = ','.join(tags)
    tags_string = tags_string.lower()
    tags_string = cond_lemmatize(tags_string)
    tags_string = tags_string.replace(' ', '_')
    tags_string = tags_string.replace('/', ' ')
    return tags_string

def extract_tags(records):
    for r in records:
        tags = r["tags"]
        categories = r["categories"]
        tags.extend(categories)
        tags = list(set(tags))
        tags = format_tags(tags)

def get_freq(org_df, fn, dump=False):
    cv = CountVectorizer(strip_accents='unicode', stop_words='english')
    tc = cv.fit_transform(org_df['annotation'])
    frequencies = sum(tc).toarray()[0]
    freq = pd.DataFrame(frequencies, index=cv.get_feature_names(), columns=['frequency'])
    freq.sort_values(['frequency'], ascending=[0], inplace=True)
    if dump == True:
        freq.to_csv(fn)

def cluster(model, tag_dataframe):
    count_model_included = 0
    count_model_nonincluded = 0
    venue_vectors = []
    for idx,row in tag_dataframe.iterrows():
        tags = row['annotation'].split()
        word_vectors = np.empty(shape=(len(tags), 300))
        idx_word_vectors = 0
        for tag in tags:
            tag = tag.replace('_', ' ')
            if tag in model.vocab:
                word_vectors[idx_word_vectors] = model[tag]
                idx_word_vectors += 1
            else:
                tokens = tag.split()
                token_vectors = np.empty(shape=(len(tokens), 300))
                idx_token_vectors = 0
                for token in tokens:
                    if token in model.vocab:
                        token_vectors[idx_token_vectors] = model[token]
                        idx_token_vectors += 1
                    else:
                        # print token
                        continue
                if idx_token_vectors > 0:
                    word_vectors[idx_word_vectors] = np.average(token_vectors[:idx_token_vectors], axis=0)
                    idx_word_vectors += 1
        
        if idx_word_vectors != 0 or idx_token_vectors != 0:
            count_model_included += 1
            venue_vectors.append(np.average(word_vectors[:idx_word_vectors], axis=0))
            # tag_dataframe.set_value(idx, 'vector_rep', np.average(word_vectors[:idx_word_vectors], axis=0))
        else:
            count_model_nonincluded += 1
#            print(idx)
            venue_vectors.append(np.nan)
            # tag_dataframe.set_value(idx, 'vector_rep', np.nan)
    tag_dataframe['vector_rep'] = venue_vectors
    tag_dataframe = tag_dataframe.dropna(subset=['vector_rep'], how='any')
    reduced_TV = np.matrix(tag_dataframe['vector_rep'].tolist())
    #reduced_TV = tag_dataframe['vector_rep']
    
    print reduced_TV
    print type(reduced_TV)
    print(count_model_included, ", ", count_model_nonincluded)

if __name__ == "__main__":

    print "train model"
#    model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join("data","GoogleNews-vectors-negative300.bin"), binary=True)
#    model.save(os.path.join(output_path_files,"word2vec.model")) 

    model  = gensim.models.KeyedVectors.load(os.path.join(output_path_files,"word2vec.model")) 

    f = open('london.venues.json','rb')
    records = json.load(f)
    df = pd.DataFrame(records)
    column_to_analyze = ['categories','tags']
    for column in column_to_analyze:
        df[column] = df[column].apply(format_tags)

    df['tags'] = df['tags'].apply(clean_noise)
    df['categories'] = df['categories'].apply(clean_noise)
    df['annotation'] = df[['categories', 'tags']].apply(lambda x: ' '.join(x), axis=1)
    get_freq(df, 'tag_frequency.csv', dump=True)
#    with open('venue_tags.pkl', 'wb') as vt:
#        cPickle.dump(df, vt, 2)

    cluster(model, df)


