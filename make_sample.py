import pandas as pd

IN = r'd:\Sarcasm Detection\train-balanced-sarcasm.cleaned.csv'
OUT = r'd:\Sarcasm Detection\train_sample.csv'
N = 2000

print('Reading', IN)
df = pd.read_csv(IN, engine='python')
if 'label' not in df.columns or 'text' not in df.columns:
    raise SystemExit('Expected columns `text` and `label` in cleaned CSV')

# stratified sample if possible
try:
    sampled = df.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=min(1, N/len(df)), random_state=42))
    if len(sampled) > N:
        sampled = sampled.groupby('label', group_keys=False).apply(lambda x: x.sample(n=max(1, int(N/len(sampled)*len(x))), random_state=42))
except Exception:
    sampled = df.sample(n=min(N, len(df)), random_state=42)

sampled = sampled.reset_index(drop=True)
sampled.to_csv(OUT, index=False)
print('Wrote sample to', OUT, 'rows=', len(sampled))
