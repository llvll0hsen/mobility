import json
from collections import defaultdict,Counter
from operator import itemgetter 
import sys

import matplotlib.pylab as plt
import googlemaps
from google_account import account

app = account['mohsen']
gmaps = googlemaps.Client(key = app['api_key'])

def create_lat_lon_file(records):
    with open('loc_bcn.txt','wb') as f:
        for i in records:
            lat,lon = i['geolocation']['coordinates']
            city = (i['address']['city'])
            if city:
                if city == 'Barcelona':

                #if (city in ['Barcellona','Barcelna']) or ('Barcelona' in city):
                    f.write('\n{0} {1}'.format(lon,lat))


def with_ratings(records):
    source_ratings = defaultdict(int)
    source_loc_with_tag = defaultdict(set)
    source_count = defaultdict(float)
    for r in records:
        prices = r['ratings']
        name = r['name']
        for source,rating_dict in prices.iteritems():
            source_count[source]+=1.
            rating = rating_dict['rating']
            if rating:
                source_loc_with_tag[source].add(name)
                source_ratings[source]+=1
    per_source_names = source_loc_with_tag.values()
    intersections = set.intersection(*per_source_names)
    source_ratings['overlapped'] = len(intersections)
    
    source_ratings2 = {k:source_ratings[k]/v for k,v in source_count.iteritems()}
    temp  = sorted(source_ratings2.items(),key=itemgetter(1))
    source, counts = zip(*temp)
    #print counts
    fig,ax = plt.subplots()
    x = range(len(source))
    ax.bar(x,counts, align="center")
    ax.set_xticks(x)
    ax.set_xticklabels(source,rotation=45)
    plt.savefig('ratings_ratio.pdf',bbox_inches='tight')
    plt.close()


def with_price_tag(records):
    colors = ['red','blue','green','black'] 
    source_tag_count = defaultdict(int)
    source_loc_with_tag = defaultdict(set)
    source_count = defaultdict(float)
    for r in records:
        prices = r['price']
        name = r['name']
        for source, p_tag in prices.iteritems():
            source_count[source]+=1.
            if p_tag:
                source_loc_with_tag[source].add(name)
                source_tag_count[source][len(p_tag)]+=1
    
    per_source_names = source_loc_with_tag.values()
    intersections = set.intersection(*per_source_names)
    source_tag_count['overlapped'] = len(intersections)

    #source_tag_count2 = {k:source_tag_count[k]/v for k,v in source_count.iteritems()}
    temp = sorted(source_tag_count.items(),key=itemgetter(1))
    source, counts = zip(*temp)
    #print counts
    fig,ax = plt.subplots()
    x = range(len(source))
    ax.bar(x,counts, align="center")
    ax.set_xticks(x)
    ax.set_xticklabels(source,rotation=45)
    plt.savefig('price_tags.pdf',bbox_inches='tight')
    plt.close()

def source_cat_count(records):
    source_cat = defaultdict(Counter)
    categories_set = set()
    for r in records:
        categories = r['categories']
        categories_set.update(set(categories))
        providers = r['providers']
        for p in providers:
            source_cat[p[0]].update(Counter(categories))
    print 'number of cats: ', len(categories_set)
    for provider, cats in source_cat.iteritems():
        cats_lim = [k for k,v in cats.iteritems() if v>9]
        cats = sorted(cats.items(),key=itemgetter(1),reverse=True)[:30]
        print provider, len(cats_lim)
        fig,ax = plt.subplots()
        loc, count = zip(*cats)
        x = range(len(loc))
        ax.bar(x,count, align="center")
        ax.set_xticks(x)
        ax.set_xticklabels(loc,rotation=90)
        plt.savefig('{0}.pdf'.format(provider),bbox_inches='tight')
        plt.close()
        
if __name__ == '__main__':
    f = open('geosegmentation_venues_170904.json','rb')
    records = json.load(f)
    #source_cat_count(records)
    #with_price_tag(records)
    #with_ratings(records)
    create_lat_lon_file(records)


