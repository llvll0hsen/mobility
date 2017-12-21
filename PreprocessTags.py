import re
import sys
import ast
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

wnl = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')
cachedStopWords = stopwords.words("english")

def getLong(lonlat):
#   long_lat = ast.literal_eval(lonlat)
#   return long_lat[0]
#    return lonlat[0]
    return "{0:.6f}".format(lonlat[0])

def getLat(lonlat):
#   long_lat = ast.literal_eval(lonlat)
#   return long_lat[1]
#    return lonlat[1]
    return "{0:.6f}".format(lonlat[1])

def mergeRedundantVenues(OrgDF, RedundantVenues):
  VenuesToDrop = []
  for idx, row in RedundantVenues.iterrows():
    VIDs = row['VID'].split(",")
    Tags = []
    MergeTarget = []
    for VID_idx in range(0, len(VIDs)):
      if VIDs[VID_idx] not in OrgDF.index:
        continue
      if len(MergeTarget) > 0:
        VenuesToDrop.append(VIDs[VID_idx])
      else:
        MergeTarget.append(VIDs[VID_idx])
      R_Ven = OrgDF.loc[VIDs[VID_idx], 'annotation']
      Tags.append(R_Ven)
    if len(MergeTarget) > 0:
      OrgDF.loc[MergeTarget[0], 'annotation'] = ' '.join(Tags)
      del MergeTarget[:]
  OrgDF.drop(VenuesToDrop, inplace=True)

def cond_lemmatize(tokens):
    if isinstance(tokens, str) or isinstance(tokens,unicode):
        tags = re.split('/|,', tokens)
        l_tags = []
        for tag in tags:
            words = tokenizer.tokenize(tag)
#            words = tag.split()
            l_words = []
            for word in words:
                if re.match("^4|5", word) and len(word) == 24:
                    continue
                if word.isdigit() and int(word) > 2016:
                    continue
                l_words.append(wnl.lemmatize(word))
            if len(l_words) == 0:
              continue
            l_tags.append(' '.join(l_words))
        return '/'.join(l_tags)
    else:
        return None

def lemmatize(tokens):
  tags = tokens.split('/')
  l_tags = []
  for tag in tags:
    words = tokenizer.tokenize(tag)
    l_words = []
    for w in words:
      w_l = w.lower()
      l_words.append(wnl.lemmatize(w_l))
    l_tags.append('_'.join(l_words))
  return ' '.join(l_tags)

def count_meta(tokens):
    if isinstance(tokens, str):
        return int(len(tokens.split()))
    else:
        return None

def extract_oid(ObjectItem):
    return ObjectItem['$oid']

def extract_oid2(ObjectString):
    ObjectItem = ast.literal_eval(ObjectString);
    return ObjectItem['$oid']

##### Pre-processing for term-vector weighting: add unigrams of multi word tags
def expand_tokens(tags):
  tag_list = tags.split()
  for tag in tag_list:
    if '_' in tag:
      for subtag in tag.split('_'):
        if subtag in cachedStopWords or subtag.isdigit():
          continue
        tag_list.append(subtag)

  return ' '.join(tag_list)
