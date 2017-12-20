import os
import sys
import csv

from bokeh.io import show, export_png
from bokeh.models import (
            ColumnDataSource,
            HoverTool,
            LogColorMapper,
            LinearColorMapper,
            ColorBar,
            LogTicker)
from bokeh.palettes import Blues9 as palette
from bokeh.plotting import figure,save
import numpy as np
import pylab as plt 
import pygeoj
import pandas as pd

from util import output_path_files, output_path_plots, census_data_fpath

def get_values(census_name):
    fpath = os.path.join(output_path_files,"census_{0}_coords_vals.txt".format(census_name))
    f = open(fpath,"rb")
    barris_val_dict = {}
    for row in f.readlines()[1:]:
        barris, val = row.strip().lower().split(";")
        barris_val_dict[barris] = float(val)
    return barris_val_dict

def load_lsoa_polygons(census_name):
    polygon_data = pygeoj.load(filepath="london2.geojson")
#    polygon_data = pygeoj.load(filepath="barris_geo_org.json")
#    polygon_data = pygeoj.load(filepath="LondonLSOA.geojson")
    spatial_entity_to_coordinates = {}
#    spatial_entity_to_values = get_values(census_name)
    spatial_entity_to_values = {} 
    for feature in polygon_data:
        coords = feature.geometry.coordinates
        coords = coords[0]
#        print coords
#        print "----------------"

#        lsoa_id = feature.properties['N_Barri'].lower().encode("utf-8")
        lsoa_id = feature.properties['name'].lower()
        try:
            xs = [i for i,j in coords]
            ys = [j for i,j in coords]
        except Exception as err:
            coords = coords[0]
            xs = [i for i,j in coords]
            ys = [j for i,j in coords]

        spatial_entity_to_coordinates[lsoa_id] = [xs,ys]
        spatial_entity_to_values[lsoa_id] = [np.random.rand()+0.1]
#    print zip(sorted(spatial_entity_to_coordinates.keys()),sorted(spatial_entity_to_values.keys()))
#    for k,v in  spatial_entity_to_values.items():
#        print k,v
    return spatial_entity_to_coordinates,spatial_entity_to_values
 
def plot_bokeh_intensity_map(spatial_entity_to_coordinates, spatial_entity_to_values, census_name):
        census_name = census_name.replace("_"," ")
        entity_xs = []
        entity_ys = []
        entity_names = []
        entity_rates = []
        for name, coords in spatial_entity_to_coordinates.items():
            xs = [i for i in coords[0]]
            ys = [i for i in coords[1]] 
            try:
                intensity_value = spatial_entity_to_values[name]
#                intensity_value = np.median(spatial_entity_to_values[name])
            except Exception as err:
#                print "err"
#                print name
                continue

            entity_xs.append(xs)
            entity_ys.append(ys)
            entity_names.append(name)
            entity_rates.append(intensity_value)

#        palette.reverse()
        vals = spatial_entity_to_values.values()
#        color_mapper = LogColorMapper(palette=palette, low=0,high=1)
        color_mapper = LinearColorMapper(palette=palette, low=min(vals),high=max(vals))

        source = ColumnDataSource(data=dict(
            x=entity_xs,
            y=entity_ys,
            name=entity_names,
            rate=entity_rates,))

#        color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(),
#                                     label_standoff=12, border_line_color=None, location=(0,0))
        color_bar = ColorBar(color_mapper=color_mapper, 
                                     label_standoff=12, border_line_color=None, location=(0,0))
        TOOLS = "pan,wheel_zoom,reset,hover,save"
#        TOOLS = "hover"

        p = figure(
            title=census_name, tools=TOOLS,
            x_axis_location=None, y_axis_location=None,plot_width=1000)
        p.grid.grid_line_color = None

        p.patches('x', 'y', source=source,
                  fill_color={'field': 'rate', 'transform': color_mapper},
                  fill_alpha=0.7, line_color="white", line_width=0.5)
        
        hover = p.select_one(HoverTool)
        hover.point_policy = "follow_mouse"
        hover.tooltips = [
            ("Name", "@name"),
#            ("Price change rate)", "@rate%"),
            (census_name, "@rate"),
            ("(Long, Lat)", "($x, $y)"),]

        p.add_layout(color_bar, 'right')
#        export_png(p, os.path.join(output_path_plots,"census","london","{0}.png".format(census_name)))
#        fname = os.path.join(output_path_plots,"interactive_maps","london","{0}.html".format(census_name))
#        export_png(p, os.path.join(output_path_plots,"mobility","antennas","{0}.png".format(census_name)))
        fname = os.path.join(output_path_plots,"mobility","antennas","{0}.html".format(census_name))
        save(obj=p,title=census_name ,filename=fname)
#	show(p)

def remove_invald_regions(df):
    a = ["England", "Inner London", "United Kingdom","Outer London"]
    df = df[~df["Area name"].isin(a)]
    print df["Area name"].unique()
    return df

if __name__ == "__main__":
    census_name = sys.argv[1]
    col_names = ["Area name","GLA Population Estimate 2017","GLA Household Estimate 2017","Average Age, 2017",
            "Proportion of population aged 0-15, 2015", "Proportion of population of working-age, 2015",
            "Proportion of population aged 65 and over, 2015", "Net internal migration (2015)", "Net international migration (2015)",
            "Net natural change (2015)","% of resident population born abroad (2015)", "Overseas nationals entering the UK (NINo), (2015/16)",
            "Employment rate (%) (2015)","Unemployment rate (2015)","Youth Unemployment (claimant) rate 18-24 (Dec-15)",
            "Proportion of the working-age population who claim out-of-work benefits (%) (May-2016)","% working-age with a disability (2015)",
            "Gross Annual Pay, (2016)", "Jobs Density, 2015","Number of jobs by workplace (2014)","Number of active businesses, 2015",
            "Crime rates per thousand population 2014/15", "Median House Price, 2015", "Homes Owned outright, (2014) %","% of area that is Greenspace, 2005",
            "Number of cars, (2011 Census)","Average Public Transport Accessibility score, 2014","Rates of Children Looked After (2016)",
            "Achievement of 5 or more A*- C grades at GCSE or equivalent including English and Maths, 2013/14",
            "% of pupils whose first language is not English (2015)","% children living in out-of-work households (2015)",
            "Happiness score 2011-14 (out of 10)","Worthwhileness score 2011-14 (out of 10)","Anxiety score 2011-14 (out of 10)",
            "Childhood Obesity Prevalance (%) 2015/16","Political control in council","Proportion of seats won by Conservatives in 2014 election",
            "Proportion of seats won by Labour in 2014 election", "Proportion of seats won by Lib Dems in 2014 election","Turnout at 2014 local elections"
            ]
#    col_names = ["Area name","Gross Annual Pay, (2016)"]
    df = pd.read_excel(os.path.join(census_data_fpath,"london","london-borough-profiles.xlsx"),sheetname="Data")
    df = df[col_names].dropna()
    df = remove_invald_regions(df)

    del col_names[0]

    
    for c in col_names:
        spatial_entity_to_values = {}
        temp = df[["Area name",c]]
        for row in temp.itertuples():
            area_name, value = row[1],row[2]
            spatial_entity_to_values[area_name.lower()] = value
           
        census_name = c.replace("/","-")
        a, b =  load_lsoa_polygons(census_name)
        
#        print len(spatial_entity_to_values.keys())
        plot_bokeh_intensity_map(a,spatial_entity_to_values,census_name)

