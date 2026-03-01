import pandas as pd
from pathlib import Path
PROC_DIR = Path('data/processed')
data_path = PROC_DIR / 'combined_laps.csv'
raw = pd.read_csv(data_path)
print('Team in columns?', 'Team' in raw.columns)
print('Team unique values:', raw.get('Team', pd.Series(['N/A'])).unique())
