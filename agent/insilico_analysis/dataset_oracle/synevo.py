import pandas as pd
import numpy as np
from typing import Dict, List, Set, Optional

_AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


def _normalize_seq(s: str) -> str:
    return s.replace("*", "")


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


class Oracle:
    def __init__(self, csv_path: str, wt_sequence: str, value_col: str = "fitness", seq_col: str = "sequence", noise_std: float = 0.0, n_replicates: int = 1, extra_value_cols: Optional[List[str]] = None):
        # Accept CSV, TSV, or PKL transparently
        if csv_path.endswith('.pkl') or csv_path.endswith('.pickle'):
            self.df = pd.read_pickle(csv_path)
        elif csv_path.endswith('.tsv') or csv_path.endswith('.txt'):
            self.df = pd.read_csv(csv_path, sep='\t')
        else:
            self.df = pd.read_csv(csv_path)
        self.seq_col = seq_col
        self.value_col = value_col
        self.noise_std = float(noise_std)
        self.n_replicates = int(n_replicates)
        self.wt = _normalize_seq(wt_sequence)
        seqs = self.df[self.seq_col].astype(str).str.replace("*", "")
        # Main map: only sequences with a non-NaN main value (acquirable)
        vals = pd.to_numeric(self.df[self.value_col], errors='coerce')
        main = pd.Series(vals.values, index=seqs).dropna()
        self._map: Dict[str, float] = main.to_dict()
        # Extra maps: per-column lookup, NaN preserved as missing
        self._extra_maps: Dict[str, Dict[str, float]] = {}
        for col in (extra_value_cols or []):
            if col == self.value_col or col not in self.df.columns:
                continue
            ev = pd.to_numeric(self.df[col], errors='coerce')
            extra = pd.Series(ev.values, index=seqs).dropna()
            self._extra_maps[col] = extra.to_dict()

    def query(self, sequences: List[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for s in sequences:
            key = _normalize_seq(s)
            if key not in self._map:
                # skip invalid sequences (not in dataset, or missing main value)
                continue
            out[s] = float(self._map[key])
        return out

    def lookup_extra(self, sequence: str, col: str) -> Optional[float]:
        """Return the value for an auxiliary (non-main) target column, or None if missing."""
        key = _normalize_seq(sequence)
        m = self._extra_maps.get(col)
        if m is None:
            return None
        v = m.get(key)
        return float(v) if v is not None else None


def insilico_cloud_lab_acquisition(segmented_sequences: List[List[str]], oracle: Oracle):
    sequences = [''.join(segs) for segs in segmented_sequences]
    phenos = oracle.query(sequences)
    invalid = [s for s in sequences if s not in phenos]
    phenotype_data = {seq: [phenos[seq]] for seq in phenos}
    return phenotype_data, invalid


def insilico_get_dict_from_cloud_lab_acquisition(segmented_sequences: List[List[str]], assayed_target_name: List[str], target_to_index_mapping: Dict[str, int], oracle: Oracle):
    phenotype_data, invalid_sequences = insilico_cloud_lab_acquisition(segmented_sequences, oracle)
    out = {}
    for segs in segmented_sequences:
        seq = ''.join(segs)
        if seq in invalid_sequences:
            continue
        out[seq] = {}
        for target in assayed_target_name:
            out[seq][target] = phenotype_data[seq][target_to_index_mapping[target]]
    return out, invalid_sequences


class Frontier:
    def __init__(self, dataset_csv: str, seq_col: str = "sequence", value_col: str = "fitness", restrict_length: Optional[int] = None, wt_sequence: Optional[str] = None):
        # Accept CSV, TSV, or PKL transparently
        if dataset_csv.endswith('.pkl') or dataset_csv.endswith('.pickle'):
            df = pd.read_pickle(dataset_csv)
        elif dataset_csv.endswith('.tsv') or dataset_csv.endswith('.txt'):
            df = pd.read_csv(dataset_csv, sep='\t')
        else:
            df = pd.read_csv(dataset_csv)
        self.seq_col = seq_col
        self.value_col = value_col
        # Filter to rows with a non-NaN main value so the frontier only contains acquirable seqs.
        # No-op for single-target datasets where value_col is always populated.
        if value_col in df.columns:
            df = df[pd.to_numeric(df[value_col], errors='coerce').notna()].copy()
        all_seqs = df[self.seq_col].astype(str).str.replace("*", "")
        if restrict_length is not None:
            all_seqs = all_seqs[all_seqs.str.len() == int(restrict_length)]
        seq_set = set(all_seqs)
        # Inject WT into the sequence set if not already present
        if wt_sequence is not None:
            wt_norm = _normalize_seq(wt_sequence)
            seq_set.add(wt_norm)
        self.all: List[str] = sorted(seq_set)
        self._set: Set[str] = set(self.all)
        # Cache length: use restrict_length if provided, else infer from data
        self._L: int = int(restrict_length) if restrict_length is not None else (len(self.all[0]) if self.all else 0)

    def neighbors_of(self, seq: str) -> List[str]:
        seq_norm = _normalize_seq(seq)
        if self._L == 0:
            return []
        assert len(seq_norm) == self._L, "Sequence length mismatch"
        neighbors = []
        s_list = list(seq_norm)
        for i in range(self._L):
            original = s_list[i]
            for aa in _AA_ALPHABET:
                if aa == original:
                    continue
                s_list[i] = aa
                cand = ''.join(s_list)
                if cand in self._set:
                    neighbors.append(cand)
            s_list[i] = original
        return neighbors

    def initial_labeled(self, seeds: List[str]) -> Set[str]:
        return set(_normalize_seq(s) for s in seeds if _normalize_seq(s) in self._set)

    def frontier(self, labeled: Set[str]) -> List[str]:
        F: Set[str] = set()
        for s in labeled:
            for t in self.neighbors_of(s):
                if t not in labeled:
                    F.add(t)
        return sorted(F)

    def frontier_k(self, labeled: Set[str], k: int = 1) -> List[str]:
        """All dataset sequences within Hamming distance <= k of any labeled seq, excluding labeled.
        Uses multi-layer neighbor expansion over the dataset graph.
        """
        if k <= 1:
            return self.frontier(labeled)
        base = set(_normalize_seq(s) for s in labeled)
        visited: Set[str] = set(base)
        frontier_set: Set[str] = set()
        current_layer: Set[str] = set(base)
        for _ in range(k):
            next_layer: Set[str] = set()
            for s in current_layer:
                for t in self.neighbors_of(s):
                    if t in visited:
                        continue
                    visited.add(t)
                    next_layer.add(t)
                    if t not in base:
                        frontier_set.add(t)
            current_layer = next_layer
            if not current_layer:
                break
        return sorted(frontier_set)

    def neighbors_at_exact_distance(self, seq: str, k: int) -> List[str]:
        """Return all dataset sequences at exactly Hamming distance k from seq.
        Uses BFS layers over the dataset neighbor graph. Returns empty for k < 1.
        """
        seq_norm = _normalize_seq(seq)
        if k < 1 or self._L == 0:
            return []
        assert len(seq_norm) == self._L, "Sequence length mismatch"
        visited: Set[str] = {seq_norm}
        current_layer: Set[str] = {seq_norm}
        for _ in range(k):
            next_layer: Set[str] = set()
            for s in current_layer:
                for t in self.neighbors_of(s):
                    if t in visited:
                        continue
                    visited.add(t)
                    next_layer.add(t)
            current_layer = next_layer
            if not current_layer:
                break
        # After k expansions, current_layer holds exactly distance-k nodes
        return sorted(current_layer)

    @staticmethod
    def df_from_seqs(seqs_with_star: List[str]) -> pd.DataFrame:
        return pd.DataFrame({
            'sequence': seqs_with_star,
            'segmented_sequence': [[s] for s in seqs_with_star]
        })
