import pandas as pd
import os

movie_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/movie_data.csv'))

