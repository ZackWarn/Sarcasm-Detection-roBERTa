import pandas as pd
import sys

path = r'd:\Sarcasm Detection\train-balanced-sarcasm.cleaned.csv'
if len(sys.argv) > 1:
    path = sys.argv[1]

try:
    df = pd.read_csv(path, engine='python')
except Exception as e:
    print('Failed to read CSV:', e)
    raise SystemExit(1)

print('Columns:', df.columns.tolist())
print('\nClass distribution:')
print(df['label'].value_counts().to_dict())
print('\nSample rows:')
print(df.head(5).to_string())
