import json
from collections import defaultdict,Counter
from operator import itemgetter 
import sys
from datetime import datetime
import os

import requests
import numpy as np
import matplotlib.pylab as plt
import googlemaps
from wordcloud import WordCloud
import dill
from pymongo import MongoClient
import pandas as pd

from google_account import account
from util import output_path_files, output_path_plots,connect_monogdb, reverse_geo_mongodb

app = account['mohsen']
gmaps = googlemaps.Client(key = app['api_key'])


def reverse_geo_osm(lat,lon):
    api_req = "http://nominatim.openstreetmap.org/reverse?format=json&lat={0}&lon={1}&zoom=18&addressdetails=1".format(lat,lon)
    r = requests.get(api_req)
    if r.ok:
        rdata = json.loads(r.content)
        print rdata
        if "city_district" in rdata["address"]:
            result =rdata["address"]["city_district"].lower().encode("utf-8")
        else:
            result = None
    else:
        result = None
    return result

def price_cat_loc(records):
    source_tag_loc = defaultdict(lambda: defaultdict(list))
#    source_loc_with_tag = defaultdict(set)
#    source_count = defaultdict(float)
    for r in records:
        prices = r['price']
        name = r['name']
        loc = r['geolocation']['coordinates']
        ns = 0.
        n_tag = 0
        for source, p_tag in prices.iteritems():
            if p_tag:
                ns+=1
                try:
                    ptag = len(p_tag)
                except Exception as err: 
                    #for google
                    ptag = p_tag
                n_tag += ptag
                source_tag_loc[source][ptag].append(loc)
        if ns:
            source_tag_loc['avg'][int(n_tag/ns)].append(loc)
    
        for source, star_locs in source_tag_loc.iteritems():
            for star, locs in star_locs.iteritems():
                fpath = os.path.join(output_path_files,'{0}_{1}.txt'.format(source,star)) 
                with open(fpath,'wb') as f:
                    f.write('lon,lat')
                    for l in locs:
                        f.write('\n{0},{1}'.format(l[0],l[1]))

def neighborhood_price_dist(records):
    collection,client = connect_monogdb()
    f = open(os.path.join(output_path_files,"dist_ne_price.txt"),"wb")
    f.writelines("lon;lat;district;neighborhood;avg_price")
    neighborhoods = set()
    neighborhoods_price = defaultdict(lambda: defaultdict(int))
    for r in records:
        lon,lat = r['geolocation']['coordinates']
        prices = r['price']
        ne = reverse_geo_mongodb(lat,lon,collection)
        if ne:
            p_tags = []
            for p in prices.values():
                if p is not None:
                    try:
                        p_tags.append(len(p))
                    except Exception as err:
                        #for google
                        p_tags.append(p)
            if p_tags:
                print p_tags
                f.writelines("\n{0};{1};{2};{3};{4}".format(lon,lat,ne[0],ne[1],float(sum(p_tags))/len(p_tags)))
    f.close()
    client.close()

def plot_neighborhood_price_dist():
    f = open(os.path.join(output_path_files,"dist_ne_price.txt"),"rb")
    df = pd.read_csv(f, delimiter = ";")
    dist_group = df.groupby("neighborhood")["avg_price"]
    fig,ax = plt.subplots()
    districts = dist_group.groups.keys()
    data = []
    for i,district in enumerate(districts):
        vals = np.array(dist_group.get_group(district))
        districts[i] =  district.decode("utf-8")
        data.append(vals)

    ax.boxplot(np.array(data),labels=districts)
    xtickNames = plt.setp(ax, xticklabels=districts)
    plt.setp(xtickNames, rotation=90, fontsize=8)

    ax.tick_params(direction='out')
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False) 
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    fpath = os.path.join(output_path_plots,'{0}_avg_price.pdf'.format("ne"))
    plt.savefig(fpath,bbox_inches='tight')
    plt.close()
        

def categories_with_hour(records):
    cats  = []
    for r in records:
        cats_temp = r["categories"]
        time = r["opening_hours"]
        if time:
            cats.extend(cats_temp)
    #print cats
    cats = Counter(cats)
    del cats["point_of_interest"]
    del cats["establishment"]
    cats = sorted(cats.items(),key=itemgetter(1),reverse=True)[:50]
    
    fig,ax = plt.subplots()
    loc, count = zip(*cats)
    x = range(len(loc))
    ax.bar(x,count, align="center")
    ax.set_xticks(x)
    ax.set_xticklabels(loc,rotation=90)
    ax.tick_params(direction='out')
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False) 
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    fpath = os.path.join(output_path_plots,'cat_dist_with_opening_hours.pdf')
    plt.savefig(fpath,bbox_inches='tight')
    plt.close()

def create_lat_lon_file(records):
    with open('loc.txt','wb') as f:
        for i in records:
            lat,lon = i['geolocation']['coordinates']
            city = (i['address']['city'])
            if city:
                if city == 'Barcelona':

                #if (city in ['Barcellona','Barcelna']) or ('Barcelona' in city):
                    f.write('\n{0} {1}'.format(lon,lat))

def tag_clouds(records):
    tags_list = []
    for r in records:
        tags = r['tags']
        tags_list.extend(tags)
    print len(tags_list)
    tags = ' '.join(tags_list)
    wordcloud = WordCloud(width=600, height=500, min_font_size=8).generate(tags)
    wordcloud.to_file("tagscloud.png")

def time_dist(records):
    source_time_dist = defaultdict(Counter)
    for r in records:
        opening_hours = r['opening_hours']
        if opening_hours:
            for source, time_records in opening_hours.iteritems():
                if time_records[0]:
                    slots = set()
                    for tr in time_records:
                        if tr:
                            print tr
                            temp = tr[0]
                            topen = datetime.strptime(temp['open'], "%H:%M").timetz().hour
                            tclose = datetime.strptime(temp['close'], "%H:%M").timetz().hour
                            s = "{0}-{1}".format(topen,tclose)
                            if s not in slots:
                                slots.add(s)
                                source_time_dist[source].update(range(topen,tclose))
    for source, time_dist in source_time_dist.iteritems():
        fig,ax = plt.subplots()
        ax.hist(list(time_dist.elements()),15,normed=True, alpha=0.75)
        ax.set_title(source)
        ax.tick_params(direction='out')
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False) 
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
        plt.savefig(os.path.join(output_path_plots,'time_dist_{0}.pdf'.format(source)),bbox_inches='tight')
        plt.close()

def with_hours(records):
    collection,client = connect_monogdb()
    f = open(os.path.join(output_path_files,"hours_geo_new.txt"),"wb")
    f.writelines("\nlon;lat;district;topen;tclose")

    count_valid = 0
    source_valid = defaultdict(int)
    
    n = len(records)

    for r in records:
        opening_hours = r['opening_hours']
        if opening_hours:
            for source, time_records in opening_hours.iteritems():
                if time_records[0]:
#                    print r
#                    print '\n'
                    lon,lat = r['geolocation']['coordinates']
                    district = reverse_geo_mongodb(lat,lon,collection)


                    count_valid+=1
                    temp = time_records[0][0]
                    topen = datetime.strptime(temp['open'], "%H:%M").timetz().hour
                    tclose = datetime.strptime(temp['close'], "%H:%M").timetz().hour
                    
                    f.writelines("\n{0};{1};{2};{3};{4}".format(lon,lat,district,topen,tclose))
#                    source_time_dist[source].update(range(topen,tclose))
                    source_valid[source]+=1
    f.close()
    temp = sorted(source_valid.items(), key = itemgetter(1))
    sources, counts = zip(*temp)
    counts = np.array(counts, dtype='float')
    counts = counts/n
    fig,ax = plt.subplots()
    x = range(len(sources))
    ax.bar(x,counts, align="center")
    ax.set_xticks(x)
    ax.set_xticklabels(sources,rotation=45)
    plt.savefig(os.path.join(output_path_plots,'openings_ratio.pdf'),bbox_inches='tight')

    plt.close()
    print 'number/ratio of records with hours: ',count_valid,count_valid/float(n)
    client.close()
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
    source_tag_count = defaultdict(lambda: defaultdict(int))
    source_loc_with_tag = defaultdict(set)
    source_count = defaultdict(float)
    n = len(records)

    for r in records:
        prices = r['price']
        name = r['name']
        check_in = {"facebook":False,"google":False,"foursquare":False}
        for source, p_tag in prices.iteritems():
            source_count[source]+=1.
            if p_tag:
                if source == "google":
                    print name
                    print prices
                check_in[source] = True
                #print prices
                source_loc_with_tag[source].add(name)
                source_count[source]+=1.
                try:
                    source_tag_count[source][len(p_tag)]+=1
                except Exception as err:
                    source_tag_count[source][p_tag]+=1
        fb = check_in["facebook"]
        fs = check_in["foursquare"]
        go = check_in["google"]
        if fb==False or fs == False or go==True:
            pass

#    per_source_names = source_loc_with_tag.values()
#    intersections = set.intersection(*per_source_names)
#    source_tag_count['overlapped'] = len(intersections)

#    source_tag_count = {k:source_tag_count[k]/v for k,v in source_count.iteritems()}
    #temp = sorted(source_tag_count.items(),key=itemgetter(1))
    fig,ax = plt.subplots()
    x = range(len(source_tag_count))
    #source, counts = zip(*temp)
    sources = []
    data = np.zeros((len(x),4))
    i = 0
    for source, count_dict in source_tag_count.iteritems():
        print source, source_count[source]/n
        sources.append(source)
        stars, count = zip(*count_dict.items())
        print stars
        print count
        count = np.array(count,dtype='float')
        data[i,:] = count/sum(count)
        i+=1
     
    print data 
    ax.bar(x,data[:,0], align="center",color='green', label='*')
    ax.bar(x,data[:,1], align="center", bottom=data[:,0],color='red',label='**')
    ax.bar(x,data[:,2], align="center", bottom=data[:,0]+data[:,1],color='blue',label='***')
    ax.bar(x,data[:,3], align="center", bottom=data[:,0]+data[:,1]+data[:,2],color='grey',label='****')
    ax.set_xticks(x)
    ax.set_xticklabels(sources,rotation=45)
    ax.tick_params(direction='out')
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False) 
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    ax.legend(loc='upper center',ncol=4,frameon=True,bbox_to_anchor=(0.5, 1.2),fontsize=25)#fancybox=True)
 
    plt.savefig(os.path.join(output_path_plots,'price_tags_cat_ratio.pdf'),bbox_inches='tight')
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
        ax.set_title(provider)
        ax.tick_params(direction='out')
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False) 
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
        fpath = os.path.join(output_path_plots,'{0}_cat_dist.pdf'.format(provider))
        plt.savefig(fpath,bbox_inches='tight')
        plt.close()
        
if __name__ == '__main__':
    f = open('geosegmentation.venues.json','rb')
    records = json.load(f)
#    neighborhood_price_dist(records)
#    plot_neighborhood_price_dist()
#    categories_with_hour(records)
#    source_cat_count(records)
#    with_price_tag(records)
#    with_ratings(records)
#    create_lat_lon_file(records)
#    with_hours(records)
#    time_dist(records)
#    tag_clouds(records)
#    price_cat_loc(records)
