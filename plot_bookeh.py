import os
import sys

from bokeh.io import show, export_png
from bokeh.models import (
            ColumnDataSource,
            HoverTool,
            LogColorMapper,
            ColorBar,
            LogTicker)
from bokeh.palettes import YlOrRd9 as palette
from bokeh.plotting import figure,save
import numpy as np
import pylab as plt 
import pygeoj
import csv
import numpy as np

from util import output_path_files, output_path_plots

def get_values(census_name):

    fpath = os.path.join(output_path_files,"census_{0}_coords_vals.txt".format(census_name))
    f = open(fpath,"rb")
    barris_val_dict = {}
    for row in f.readlines()[1:]:
        barris, val = row.strip().lower().split(";")
        barris_val_dict[barris] = float(val)
    return barris_val_dict

def load_lsoa_polygons(census_name):
    polygon_data = pygeoj.load(filepath="barris_geo_org.json")
#    polygon_data = pygeoj.load(filepath="LondonLSOA.geojson")
    spatial_entity_to_coordinates = {}
    spatial_entity_to_values = get_values(census_name)
    for feature in polygon_data:
        coords = feature.geometry.coordinates
        coords = coords[0]
#        print coords
#        print "----------------"
        lsoa_id = feature.properties['N_Barri'].lower().encode("utf-8")
        try:
            xs = [i for i,j in coords]
            ys = [j for i,j in coords]
        except Exception as err:
            coords = coords[0]
            xs = [i for i,j in coords]
            ys = [j for i,j in coords]

        spatial_entity_to_coordinates[lsoa_id] = [xs,ys]
#        spatial_entity_to_values[lsoa_id] = [np.random.rand()]
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
                print "err"
                print name
                intensity_value = 0.0

            entity_xs.append(xs)
            entity_ys.append(ys)
            entity_names.append(name)
            entity_rates.append(intensity_value)

        palette.reverse()
        color_mapper = LogColorMapper(palette=palette)


        source = ColumnDataSource(data=dict(
            x=entity_xs,
            y=entity_ys,
            name=entity_names,
            rate=entity_rates,))

        color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(),
                                     label_standoff=12, border_line_color=None, location=(0,0))
        TOOLS = "pan,wheel_zoom,reset,hover,save"

        p = figure(
            title=census_name, tools=TOOLS,
            x_axis_location=None, y_axis_location=None)
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
        export_png(p, os.path.join(output_path_plots,"census","{0}.png".format(census_name)))
        save(obj=p, filename=os.path.join(output_path_plots,"interactive_maps","{0}.html".format(census_name)))
	show(p)
if __name__ == "__main__":
    census_name = sys.argv[1]
    a, b =  load_lsoa_polygons(census_name)
    plot_bokeh_intensity_map(a,b,census_name)

