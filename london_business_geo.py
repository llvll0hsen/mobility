import os

import numpy as np

from google_account import account
from util import output_path_files, output_path_plots,connect_monogdb, reverse_geo_mongodb

app = account['mohsen']
gmaps = googlemaps.Client(key = app['api_key'])



