import os
import sys

import pandas as pd
import numpy as np

if __name__ == "__main__":
    
    df_london = pd.read_csv(os.path.join(output_path_files, "user_top_anthena_london_only_economicrank_{0}_00_04.txt".format(month)))
