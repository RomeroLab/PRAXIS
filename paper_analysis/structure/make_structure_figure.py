"""Per-residue attribution of the chimera block effects, painted on the AlphaFold model.

All outputs go to ../exports/: bgl3_af3_attribution.pdb, view_attribution.cxc,
attribution_colorbar.{svg,png} (+ _minimal), supp_per_residue_attribution.{csv,xlsx}.
See README.md.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.colorbar
import matplotlib.style
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from pyfamsa import Aligner, Sequence

HERE = Path(__file__).resolve().parent
AGENT = HERE.parents[1] / 'agent'
SEG = AGENT / 'data' / 'sequence_segments.csv'
PRED = AGENT / 'scoring' / 'round_sweep' / 'predictions_round_24.csv'
AF_CIF = HERE / 'fold_bgl3' / 'fold_bgl3_model_0.cif'
ALN_CACHE = HERE / 'block_alignments.json'
EXPORTS = HERE.parent / 'exports'
EXPORTS.mkdir(exist_ok=True)
AF_OUT = EXPORTS / 'bgl3_af3_attribution.pdb'

# The model is the mature sequence, so its residue 1 is construct residue 8. Fold the
# tagged construct instead and this must become 0, or the whole map shifts by seven.
TAG = 7
SCALE = 100.0            # B-factor field carries two decimals; attribution is ~0.001-0.2
CAT = (178, 383)         # catalytic glutamates: acid/base, nucleophile
CLIP = 99                # colour limit percentile; 0.0846, just above the f0 plateau
LO, MID, HI = '#1735f5', '#d9d9d9', '#dc1e04'
CMAP = LinearSegmentedColormap.from_list('praxis_div', [LO, MID, HI])
LABEL = 'Per-residue share of block effect'

seg = pd.read_csv(SEG, index_col=0)
seg = seg.apply(lambda col: col.str.replace('*', '', regex=False).str.strip())  # stop codon
PARENTS, BLOCKS = list(seg.index), list(seg.columns)

pred = pd.read_csv(PRED, dtype={'chimera': str})
M = np.array([[int(c) for c in s] for s in pred['chimera']], dtype=np.int8)
S = 0.5 * (pred['a2'].to_numpy() + pred['a3'].to_numpy()) - pred['a1'].to_numpy()
eff = np.array([[S[M[:, i] == p].mean() - S.mean() for i in range(8)]
                for p in range(1, 7)])

# FAMSA is not deterministic on f0 (the only length-variable block) and the alignment moves
# the peak attribution by ~20%. This pins the published one; delete only to change them.
if ALN_CACHE.exists():
    alns = json.load(open(ALN_CACHE))
else:
    alns = {}
    for blk in BLOCKS:
        seqs = [Sequence(p.encode(), seg.loc[p, blk].encode()) for p in PARENTS]
        a = {s.id.decode(): s.sequence.decode()
             for s in Aligner(guide_tree='upgma').align(seqs)}
        alns[blk] = [a[p] for p in PARENTS]
    json.dump(alns, open(ALN_CACHE, 'w'), indent=1)
    print(f'wrote {ALN_CACHE.name}')

rows_out, offset = [], 0
for bi, blk in enumerate(BLOCKS):
    rows = alns[blk]
    n = len(rows[0])
    cols, idx = [], {}
    for j in range(n):
        for a in sorted({r[j] for r in rows}):
            idx[(j, a)] = len(cols)
            cols.append((j, a))
    X = np.zeros((6, len(cols)))
    for p in range(6):
        for j in range(n):
            X[p, idx[(j, rows[p][j])]] = 1.0

    # 6 x K with K >> 6: underdetermined, so take the minimum-norm solution. Valid because
    # X has full row rank; X @ X.T is invertible here, pinv guards a degenerate block.
    beta = X.T @ (np.linalg.pinv(X @ X.T) @ eff[:, bi])
    assert np.abs(X @ beta - eff[:, bi]).max() < 1e-9

    pos = -1
    for j in range(n):
        if rows[0][j] == '-':
            continue
        pos += 1
        resnum = offset + pos + 1 - TAG
        if resnum < 1:
            continue
        k = idx[(j, rows[0][j])]
        rows_out.append({
            'resnum': resnum,
            'construct_resnum': resnum + TAG,
            'aa': rows[0][j],
            'block': blk,
            'block_position': bi + 1,
            'block_effect_p1': eff[0, bi],
            **{f'{PARENTS[p]}_residue': rows[p][j] for p in range(6)},
            'n_parents_sharing': int(X[:, k].sum()),
            'attribution': beta[k],
        })
    offset += len(seg.loc['p1', blk])

tab = pd.DataFrame(rows_out).sort_values('resnum').reset_index(drop=True)
attrib = dict(zip(tab.resnum, tab.attribution))
lim = float(np.percentile(tab.attribution.abs(), CLIP))
vmax = float(tab.attribution.abs().max())
tab['clipped_in_figure'] = tab.attribution.abs() > lim
print(f'{len(tab)} residues; range [{tab.attribution.min():+.4f}, {tab.attribution.max():+.4f}]; '
      f'limit +/-{lim:.4f} ({CLIP}th pct), {int(tab.clipped_in_figure.sum())} clipped')

# ---- AlphaFold model -> PDB ----
hdr, cif_rows, inloop = [], [], False
for line in open(AF_CIF):
    s = line.strip()
    if s.startswith('_atom_site.'):
        hdr.append(s.split('.')[1]); inloop = True; continue
    if inloop:
        if s.startswith('#') or not s:
            break
        cif_rows.append(s.split())
c = {k: n for n, k in enumerate(hdr)}

# AF3 writes pLDDT per atom; the per-residue value is the CA's.
plddt = {int(r[c['label_seq_id']]): float(r[c['B_iso_or_equiv']])
         for r in cif_rows if r[c['label_atom_id']] == 'CA'}
ca = {int(r[c['label_seq_id']]): np.array([float(r[c['Cartn_x']]), float(r[c['Cartn_y']]),
                                           float(r[c['Cartn_z']])])
      for r in cif_rows if r[c['label_atom_id']] == 'CA'}

with open(AF_OUT, 'w') as out:
    out.write(f'REMARK 300 B-FACTOR = ATTRIBUTION x {SCALE:.0f}; OCCUPANCY = pLDDT/100\n')
    for serial, r in enumerate(cif_rows, start=1):
        rn = int(r[c['label_seq_id']])
        name, elem = r[c['label_atom_id']], r[c['type_symbol']]
        an = f'{name:<4}' if len(name) == 4 else f' {name:<3}'
        out.write(f"ATOM  {serial:5d} {an} {r[c['label_comp_id']]:>3} A{rn:4d}    "
                  f"{float(r[c['Cartn_x']]):8.3f}{float(r[c['Cartn_y']]):8.3f}"
                  f"{float(r[c['Cartn_z']]):8.3f}"
                  f"{plddt[rn]/100:6.2f}{attrib[rn]*SCALE:6.2f}          {elem:>2}\n")
    out.write('END\n')
print(f'wrote {AF_OUT.name}')

# ---- ChimeraX script ----
# '#' opens a model spec in .cxc, so comments cannot be inline.
L = f'{lim*SCALE:.2f}'
(EXPORTS / 'view_attribution.cxc').write_text(f"""\
# Open from the ChimeraX GUI; --offscreen has no OpenGL on macOS.
# 'close' stops re-runs stacking duplicate models in a live session.
close
open {AF_OUT.name}

hide atoms
show cartoon
cartoon style protein modeh default

color bfactor palette {LO}:{MID}:{HI} range -{L},{L}

# Flat, not soft: shading modulates luminance, which is the channel the colour scale uses.
lighting flat
graphics silhouettes true width 1.2
set bgColor white

# Catalytic pair, off by default:
# show :{CAT[0]},{CAT[1]} atoms
# style :{CAT[0]},{CAT[1]} stick
# color :{CAT[0]},{CAT[1]} byhetero

# Compose with attribution_colorbar.svg instead of this built-in key:
# key {LO}:-{lim:.2f} {MID}:0 {HI}:+{lim:.2f} pos 0.32,0.05 size 0.36,0.025 fontSize 14

view
save attribution_chimerax.png width 2000 supersample 3 transparentBackground true
""")
print('wrote view_attribution.cxc')

# ---- colourbar ----
mpl.style.use(str(HERE.parent / 'data' / 'praxis.mplstyle'))


def draw_bar(minimal):
    fig = Figure(figsize=(4.1, 0.68))
    ax = fig.add_axes([0.16, 0.44, 0.68, 0.28])
    cb = mpl.colorbar.ColorbarBase(ax, cmap=CMAP, norm=Normalize(vmin=-lim, vmax=lim),
                                   orientation='horizontal', extend='both', extendfrac=0.07)
    cb.outline.set_linewidth(1.0)
    cb.set_ticks([-lim, 0, lim])
    if minimal:
        cb.set_ticklabels([])
        cb.set_label('')
    else:
        cb.set_ticklabels([f'{-lim:.2f}', '0', f'{lim:.2f}'])
        cb.set_label(LABEL, labelpad=4)
        ax.tick_params(length=3.5, width=1.0, pad=2)
        # True extremes past the tips, so the clipped end isn't read as the maximum.
        # x must clear the arrowheads (extendfrac).
        for x, val, ha in ((-0.105, tab.attribution.min(), 'right'),
                           (1.105, tab.attribution.max(), 'left')):
            ax.text(x, 0.5, f'{val:+.2f}', transform=ax.transAxes,
                    ha=ha, va='center', fontsize=7, color='0.35')
    return fig


for minimal, stem in ((False, 'attribution_colorbar'), (True, 'attribution_colorbar_minimal')):
    fig = draw_bar(minimal)
    for ext in ('svg', 'png'):
        fig.savefig(EXPORTS / f'{stem}.{ext}', dpi=300, bbox_inches='tight', transparent=True)
print('wrote attribution_colorbar[.svg/.png] and _minimal')

# ---- supplementary table ----
cat_xyz = np.array([ca[r] for r in CAT])
tab['plddt'] = tab.resnum.map(plddt)
tab['dist_to_catalytic_A'] = [float(np.min(np.linalg.norm(cat_xyz - ca[r], axis=1)))
                              for r in tab.resnum]
tab['attribution'] = tab.attribution.round(5)
tab['block_effect_p1'] = tab.block_effect_p1.round(4)
tab['dist_to_catalytic_A'] = tab.dist_to_catalytic_A.round(1)
tab = tab[[c for c in tab.columns if c != 'clipped_in_figure'] + ['clipped_in_figure']]

DICT = [
    ('resnum', 'Residue number in the parent-1 mature sequence; matches the AlphaFold model '
               'and PDB 1GNX.'),
    ('construct_resnum', 'Residue number in the expressed construct (adds the MHHHHHH tag).'),
    ('aa', 'Parent-1 residue at this position.'),
    ('block', 'Recombination block (f0-f7).'),
    ('block_position', 'Block position 1-8, matching the block-effect heatmap x-axis.'),
    ('block_effect_p1', 'Marginal effect of parent 1 at this block (log units). Attributions '
                        'within a block sum to this, up to columns where parent 1 has a gap.'),
    ('p1_residue .. p6_residue', 'Residue each parent contributes at this alignment column; '
                                 '"-" is an alignment gap.'),
    ('n_parents_sharing', 'How many of the six parents carry the same residue here. Where this '
                          'is 1, every such position receives an identical attribution, because '
                          'the decomposition cannot distinguish them.'),
    ('attribution', 'Minimum-norm per-residue share of the block effect (log units).'),
    ('plddt', 'AlphaFold3 pLDDT (CA). Below ~70 the position is not reliably placed; residues '
              '1-13 are the disordered N-terminus.'),
    ('dist_to_catalytic_A', 'CA distance (A) to the nearer catalytic glutamate, E178 or E383.'),
    ('clipped_in_figure', f'TRUE if |attribution| exceeds the figure colour limit ({lim:.4f}), '
                          f'so the residue saturates and its colour understates its value.'),
]

tab.to_csv(EXPORTS / 'supp_per_residue_attribution.csv', index=False)
try:
    with pd.ExcelWriter(EXPORTS / 'supp_per_residue_attribution.xlsx') as xw:
        tab.to_excel(xw, sheet_name='per-residue attribution', index=False)
        pd.DataFrame(DICT, columns=['column', 'definition']).to_excel(
            xw, sheet_name='column definitions', index=False)
    print(f'wrote supp_per_residue_attribution.csv / .xlsx '
          f'({len(tab)} residues x {tab.shape[1]} columns)')
except ModuleNotFoundError:
    print('wrote supp_per_residue_attribution.csv (xlsx skipped: openpyxl missing)')

print(f'caption: linear scale clipped at +/-{lim:.3f} ({CLIP}th pct); '
      f'{int(tab.clipped_in_figure.sum())} residues saturate, max |attribution| {vmax:.3f}')
