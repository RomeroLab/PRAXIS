import pandas as pd
import re
import itertools
import sqlite3
from typing import List, Set


def normalize_sequence(seq: str) -> str:
	"""Return a canonical sequence string (strip gaps and trailing markers)."""
	return str(seq).replace('*', '').replace('-', '').strip()


def parse_mutation_tokens(mutations: str) -> List[str]:
	"""Parse a comma-separated mutation string into tokens like 'A109D' or 'K165*'."""
	parts = [t.strip() for t in str(mutations).split(',') if t and t.strip()]
	return [t for t in parts if re.match(r'^[A-Z]\d+[A-Z\*]$', t)]


def is_accessible(tokens: List[str], observed_sets: Set[frozenset]) -> bool:
	"""Accessible if single mutant, or any (n-1) subset exists in observed sets."""
	n = len(tokens)
	if n <= 0:
		return False
	if n == 1:
		return True
	for subset in itertools.combinations(tokens, n - 1):
		if frozenset(subset) in observed_sets:
			return True
	return False


def build_observed_mutation_sets(df: pd.DataFrame, mutations_col: str = 'mutations') -> Set[frozenset]:
	"""Build a set of frozenset tokens observed in dataset mutations column."""
	sets: Set[frozenset] = set()
	for s in df.get(mutations_col, []).astype(str):
		toks = parse_mutation_tokens(s)
		if toks:
			sets.add(frozenset(toks))
	return sets


def load_results_df(csv_path: str | None, pkl_path: str | None, db_path: str | None, table_name: str | None) -> pd.DataFrame:
	"""Load a results DataFrame from CSV/PKL/SQLite, raising on missing args."""
	if csv_path:
		return pd.read_csv(csv_path)
	if pkl_path:
		return pd.read_pickle(pkl_path)
	if db_path and table_name:
		con = sqlite3.connect(db_path)
		try:
			df = pd.read_sql_query(f"SELECT * FROM {table_name}", con)
		finally:
			con.close()
		return df
	raise ValueError("Provide either --csv, --pkl, or both --db and --table") 