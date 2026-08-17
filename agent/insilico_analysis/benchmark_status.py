#!/usr/bin/env python3
"""Quick benchmark progress checker. Run from project root: python insilico_analysis/benchmark_status.py"""
import pandas as pd, glob, sqlite3
from scipy import stats

METHODS = ['full_pipeline', 'greedy', 'onehot_mlp', 'onehot_mlp_greedy', 'random']
BASE = 'insilico_analysis/dataset_oracle'

proteins = {
    'gfp': 'insilico_analysis/dataset_oracle/gfp/gfp_SeqFxnDataset.pkl',
    'gb1': 'insilico_analysis/dataset_oracle/gb1/gb1_SeqFxnDataset.pkl',
    'pab1': 'insilico_analysis/dataset_oracle/pab1/pab1_SeqFxnDataset.pkl',
    'ube4': 'insilico_analysis/dataset_oracle/ube4/ube4_SeqFxnDataset.pkl',
}

for name, pkl in proteins.items():
    odf = pd.read_pickle(pkl)
    om = dict(zip(odf['sequence'].str.replace('*', ''), odf['functional_score']))
    sc = odf['functional_score'].values
    data_dir = f'{BASE}/{name}'
    results = sorted(glob.glob(f'{data_dir}/results/*_round_log.csv'))
    dbs = sorted(glob.glob(f'{data_dir}/data/tablespace/*/*.db'))
    ck, rows = set(), []
    for f in results:
        b = f.rsplit('/', 1)[1].replace('_round_log.csv', '')
        p = b.rsplit('_seed', 1)[1]; s = p.split('_')[0]; m = '_'.join(p.split('_')[1:])
        ck.add((s, m))
        rl = pd.read_csv(f); rl['rf'] = rl['sequence'].str.replace('*', '').map(om)
        rows.append({'seed': int(s), 'method': m, 'best': rl['rf'].max(), 'pctile': stats.percentileofscore(sc, rl['rf'].max()), 'status': 'done'})
    for db in dbs:
        b = db.rsplit('/', 1)[1].replace('.db', '')
        if 'seed' not in b:
            continue
        p = b.rsplit('_seed', 1)[1]; s = p.split('_')[0]; m = '_'.join(p.split('_')[1:])
        if (s, m) in ck:
            continue
        cn = sqlite3.connect(db)
        tb = [r[0] for r in cn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite%'").fetchall()]
        if not tb:
            cn.close(); continue
        df = pd.read_sql(f'SELECT * FROM "{tb[0]}"', cn); cn.close()
        df['rf'] = df['sequence'].str.replace('*', '').map(om)
        rows.append({'seed': int(s), 'method': m, 'best': df['rf'].max(), 'pctile': stats.percentileofscore(sc, df['rf'].max()), 'status': f'~round {(len(df)-10)//10+1}/20'})
    done = sum(1 for r in rows if r['status'] == 'done')
    total = len(METHODS) * 5
    print(f'=== {name.upper()} (max={sc.max():.2f}) [{done}/{total}] ===')
    for mt in METHODS:
        for r in sorted([r for r in rows if r['method'] == mt], key=lambda x: x['seed']):
            print(f'  {mt:20s} seed={r["seed"]}  best={r["best"]:7.4f}  pctile={r["pctile"]:6.2f}%  {r["status"]}')
    print()
