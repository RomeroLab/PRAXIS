import os
import re
import sys
import json
import sqlite3

import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from scipy import stats
import plotly.io as pio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from process_plate_data import parse_plate_data

pio.templates['mono'] = go.layout.Template(
    layout=go.Layout(font=dict(family='monospace')))
pio.templates.default = 'plotly+mono'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(BASE_DIR, 'data', 'archive')
DB_PATH = os.path.join(BASE_DIR, 'data', 'database',
                        'Bgl_spec_2026_final_database.db')

ROW_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F']
SUGAR_MAP = {'glu': 'Glucose', 'xyl': 'Xylose', 'man': 'Mannose'}
SUGAR_COLS = {'Glucose': [1, 2, 3], 'Xylose': [4, 5, 6], 'Mannose': [7, 8, 9]}
CHIMERA_PALETTE = px.colors.qualitative.Set1
CREATED_BY_COLORS = {
    'seed': '#00a285',
    'agent_a1': '#5a63a6',
    'agent_a2': '#ff4470',
    'agent_a3': '#ffa600',
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_run_folders():
    folders = [
        n for n in os.listdir(ARCHIVE_DIR)
        if re.match(r'^\d{8}_', n) and n != 'Duke_final'
    ]
    return sorted(folders, reverse=True)


def load_plate_layout(folder):
    with open(os.path.join(ARCHIVE_DIR, folder, 'plate_layout.json')) as f:
        return json.load(f)


def parse_well_label(label):
    """Handle both 'ID-sugar-rep' and 'ID-rep-sugar' formats."""
    parts = label.split('-')
    if len(parts) != 3:
        return None, None, None
    cid = parts[0]
    if parts[1] in SUGAR_MAP:
        return cid, parts[1], parts[2]
    if parts[2] in SUGAR_MAP:
        return cid, parts[2], parts[1]
    return cid, None, None


def extract_chimera_ids(layout):
    ids = []
    for row in ROW_LETTERS:
        well = f'{row}1'
        if well in layout:
            cid, _, _ = parse_well_label(layout[well])
            ids.append(cid)
        else:
            ids.append(None)
    return ids


def load_phenotype(folder):
    with open(os.path.join(ARCHIVE_DIR, folder, 'phenotype.json')) as f:
        return json.load(f)


def load_timecourse(folder):
    path = os.path.join(ARCHIVE_DIR, folder, 'raw_plate_data.csv')
    fluor, tc = parse_plate_data(path)
    return fluor, tc


def compute_norm_factors(fluorescein):
    """Return {well: factor} where factor = fluorescein[well] / group_avg."""
    factors = {}
    groups = {}  # group_key -> [(well, val)]
    for well, val in fluorescein.items():
        row = well[0]
        col = int(well[1:])
        group = row + str((col - 1) // 3)
        groups.setdefault(group, []).append((well, val))
    for wells in groups.values():
        avg = np.mean([v for _, v in wells if v > 0]) or 1.0
        for w, v in wells:
            factors[w] = v / avg if v > 0 and avg > 0 else 1.0
    return factors


def load_database():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM Bgl_spec_2026_final_assayed_sequences", conn)
    conn.close()
    return df


def build_sequence_to_chimera_map():
    mapping = {}
    for folder in get_run_folders():
        try:
            layout = load_plate_layout(folder)
            phenotype = load_phenotype(folder)
            cids = extract_chimera_ids(layout)
            for i, seq in enumerate(phenotype.keys()):
                if i < len(cids) and cids[i]:
                    mapping[seq] = cids[i]
        except Exception:
            continue
    return mapping


def build_chimera_to_folder():
    """Parse archive folder names -> {chimera_id: folder_name} (most recent wins)."""
    result = {}
    for folder in reversed(get_run_folders()):
        parts = folder.split('_')
        for cid in parts[1:7]:
            result[cid] = folder
    return result


def build_chimera_to_row():
    """Scan plate_layouts -> {(chimera_id, folder): row_letter}."""
    result = {}
    for folder in get_run_folders():
        try:
            layout = load_plate_layout(folder)
            cids = extract_chimera_ids(layout)
            for i, cid in enumerate(cids):
                if cid:
                    result[(cid, folder)] = ROW_LETTERS[i]
        except Exception:
            continue
    return result


# ---------------------------------------------------------------------------
# Preload data at startup
# ---------------------------------------------------------------------------

run_folders = get_run_folders()
db_df = load_database()
seq_to_chimera = build_sequence_to_chimera_map()
chimera_to_folder = build_chimera_to_folder()
chimera_to_row = build_chimera_to_row()

db_df['chimera_id'] = db_df['sequence'].map(seq_to_chimera).fillna(
    db_df['sequence_id'].apply(lambda x: f'Seq_{x}'))
db_df = db_df.sort_values('sequence_id').reset_index(drop=True)

enzyme_table_df = db_df[['sequence_id', 'chimera_id', 'created_by',
                         'a1', 'a2', 'a3',
                         'zero_shot_fitness_predictions']].copy()
enzyme_table_df = enzyme_table_df.rename(columns={
    'a1': 'Glucose', 'a2': 'Xylose', 'a3': 'Mannose',
    'zero_shot_fitness_predictions': 'ZS Pred',
})
for col in ['Glucose', 'Xylose', 'Mannose', 'ZS Pred']:
    enzyme_table_df[col] = enzyme_table_df[col].round(2)
enzyme_table_records = enzyme_table_df.to_dict('records')

spec_table_df = enzyme_table_df[['sequence_id', 'chimera_id', 'created_by', 'ZS Pred']].copy()
glu = enzyme_table_df['Glucose']
xyl = enzyme_table_df['Xylose']
man = enzyme_table_df['Mannose']
total = glu + xyl + man
spec_table_df['Glucose'] = np.where(total > 0, glu ** 2 / total, 0).round(2)
spec_table_df['Xylose']  = np.where(total > 0, xyl ** 2 / total, 0).round(2)
spec_table_df['Mannose'] = np.where(total > 0, man ** 2 / total, 0).round(2)
spec_table_records = spec_table_df.to_dict('records')


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div(style={'fontFamily': 'monospace'}, children=[
    html.H1('Enzyme Assay Dashboard',
            style={'textAlign': 'center', 'padding': '10px'}),
    dcc.Tabs(id='main-tabs', value='tab-run', children=[
        dcc.Tab(label='Sequence Explorer', value='tab-run'),
        dcc.Tab(label='Cross-Run Aggregate', value='tab-aggregate'),
    ]),
    html.Div(id='tab-content'),
])


# ---------------------------------------------------------------------------
# Tab builders
# ---------------------------------------------------------------------------

def build_run_tab():
    table_columns = [
        {'name': 'Chimera ID', 'id': 'chimera_id'},
        {'name': 'Creator', 'id': 'created_by'},
        {'name': 'Glucose', 'id': 'Glucose', 'type': 'numeric'},
        {'name': 'Xylose', 'id': 'Xylose', 'type': 'numeric'},
        {'name': 'Mannose', 'id': 'Mannose', 'type': 'numeric'},
        {'name': 'ZS Pred', 'id': 'ZS Pred', 'type': 'numeric'},
        {'name': 'sequence_id', 'id': 'sequence_id'},
    ]
    style_conditions = [
        {
            'if': {'filter_query': f'{{created_by}} = "{cb}"',
                   'column_id': 'created_by'},
            'backgroundColor': color,
            'color': 'white',
        }
        for cb, color in CREATED_BY_COLORS.items()
    ]
    return html.Div([
        html.Div([
            html.Label('Table Values:', style={'fontWeight': 'bold', 'marginRight': '8px'}),
            dcc.RadioItems(
                id='table-mode',
                options=[
                    {'label': 'Activity', 'value': 'activity'},
                    {'label': 'Specificity', 'value': 'specificity'},
                ],
                value='activity',
                inline=True,
                inputStyle={'marginRight': '4px'},
                labelStyle={'marginRight': '16px'},
            ),
        ], style={'padding': '0 15px', 'display': 'flex', 'alignItems': 'center', 'marginBottom': '4px'}),
        html.Div([
            html.Label('Select Enzyme:', style={'fontWeight': 'bold'}),
            dash_table.DataTable(
                id='enzyme-table',
                columns=table_columns,
                hidden_columns=['sequence_id'],
                data=enzyme_table_records,
                row_selectable='single',
                selected_rows=[0],
                sort_action='native',
                style_as_list_view=True,
                page_action='none',
                style_cell={'textAlign': 'left', 'padding': '4px 8px',
                            'fontSize': '13px',
                            'fontFamily': 'monospace'},
                style_header={'fontWeight': 'bold',
                              'backgroundColor': '#e9ecef'},
                style_data_conditional=style_conditions,
                style_table={'overflowY': 'auto', 'maxHeight': '450px'},
            ),
        ], style={'padding': '15px'}),
        html.Div([
            html.Div([
                html.Label('Plot Mode:', style={'fontWeight': 'bold', 'marginRight': '8px'}),
                dcc.RadioItems(
                    id='plot-mode',
                    options=[
                        {'label': 'RAW', 'value': 'raw'},
                        {'label': 'Internal Std Adjusted', 'value': 'adjusted'},
                    ],
                    value='raw',
                    inline=True,
                    inputStyle={'marginRight': '4px'},
                    labelStyle={'marginRight': '16px'},
                ),
            ], style={'display': 'inline-flex', 'alignItems': 'center', 'marginRight': '40px'}),
            html.Div([
                html.Label('Median Emphasis:', style={'fontWeight': 'bold', 'marginRight': '8px'}),
                dcc.Checklist(
                    id='median-toggle',
                    options=[{'label': 'On', 'value': 'on'}],
                    value=[],
                    inline=True,
                    inputStyle={'marginRight': '4px'},
                ),
            ], style={'display': 'inline-flex', 'alignItems': 'center'}),
        ], style={'padding': '0 15px', 'display': 'flex', 'alignItems': 'center'}),
        dcc.Graph(id='timecourse-graph'),
        dcc.Graph(id='activity-bar-graph'),
        html.Div(id='metadata-panel', style={'padding': '15px'}),
    ])


SUGAR_OPTIONS = [
    {'label': 'Glucose', 'value': 'Glucose'},
    {'label': 'Xylose', 'value': 'Xylose'},
    {'label': 'Mannose', 'value': 'Mannose'},
]
SUGAR_TO_COL = {'Glucose': 'a1', 'Xylose': 'a2', 'Mannose': 'a3'}
SUGAR_COLOR = {'Glucose': '#5a63a6', 'Xylose': '#ff4470', 'Mannose': '#ffa600'}


def build_aggregate_tab():
    return html.Div([
        html.Div([
            html.Label('Sugar:', style={'fontWeight': 'bold', 'marginRight': '8px'}),
            dcc.RadioItems(
                id='agg-sugar-select',
                options=SUGAR_OPTIONS,
                value='Glucose',
                inline=True,
                inputStyle={'marginRight': '4px'},
                labelStyle={'marginRight': '16px'},
            ),
        ], style={'padding': '15px', 'display': 'flex', 'alignItems': 'center'}),
        dcc.Graph(id='agg-dist-graph'),
        dcc.Graph(id='agg-top-graph'),
        dcc.Graph(id='agg-scatter-graph'),
    ])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(Output('tab-content', 'children'), Input('main-tabs', 'value'))
def render_tab(tab):
    if tab == 'tab-run':
        return build_run_tab()
    return build_aggregate_tab()


@app.callback(
    [Output('enzyme-table', 'data'),
     Output('enzyme-table', 'columns')],
    Input('table-mode', 'value'),
)
def update_table_data(mode):
    if mode == 'specificity':
        cols = [
            {'name': 'Chimera ID', 'id': 'chimera_id'},
            {'name': 'Creator', 'id': 'created_by'},
            {'name': 'Glu Spec', 'id': 'Glucose', 'type': 'numeric'},
            {'name': 'Xyl Spec', 'id': 'Xylose', 'type': 'numeric'},
            {'name': 'Man Spec', 'id': 'Mannose', 'type': 'numeric'},
            {'name': 'ZS Pred', 'id': 'ZS Pred', 'type': 'numeric'},
            {'name': 'sequence_id', 'id': 'sequence_id'},
        ]
        return spec_table_records, cols
    cols = [
        {'name': 'Chimera ID', 'id': 'chimera_id'},
        {'name': 'Creator', 'id': 'created_by'},
        {'name': 'Glucose', 'id': 'Glucose', 'type': 'numeric'},
        {'name': 'Xylose', 'id': 'Xylose', 'type': 'numeric'},
        {'name': 'Mannose', 'id': 'Mannose', 'type': 'numeric'},
        {'name': 'ZS Pred', 'id': 'ZS Pred', 'type': 'numeric'},
        {'name': 'sequence_id', 'id': 'sequence_id'},
    ]
    return enzyme_table_records, cols


@app.callback(
    [Output('agg-dist-graph', 'figure'),
     Output('agg-top-graph', 'figure'),
     Output('agg-scatter-graph', 'figure')],
    [Input('agg-sugar-select', 'value')],
)
def update_aggregate(sugar):
    col = SUGAR_TO_COL[sugar]
    scolor = SUGAR_COLOR[sugar]
    other_sugars = [s for s in SUGAR_TO_COL if s != sugar]

    # --- 1. Activity Distribution (violin + strip) for selected sugar ---
    fig_dist = go.Figure()
    for cb, color in CREATED_BY_COLORS.items():
        sub = db_df[db_df['created_by'] == cb]
        if sub.empty:
            continue
        fig_dist.add_trace(go.Violin(
            x=[cb] * len(sub), y=sub[col],
            name=cb, legendgroup=cb,
            scalemode='width',
            marker_color=color, line_color=color,
            points='all', jitter=0.3, pointpos=0,
            hovertext=sub['chimera_id'],
        ))
    fig_dist.update_layout(
        title=f'Activity Distribution \u2014 {sugar}',
        yaxis_title='Activity (ln(1+mRFU/s))',
        violinmode='group', height=500,
    )

    # --- 2. Top 20 Performers by selected sugar ---
    # Keep best entry per chimera_id to avoid duplicate rows
    deduped = db_df.sort_values(col, ascending=False).drop_duplicates('chimera_id')
    top = deduped.nlargest(20, col).sort_values(col, ascending=True)
    ranked_ids = top['chimera_id'].tolist()
    fig_top = go.Figure()
    # Show selected sugar prominently, others dimmed
    for s in reversed(other_sugars):
        oc = SUGAR_TO_COL[s]
        fig_top.add_trace(go.Bar(
            y=top['chimera_id'], x=top[oc],
            name=s, orientation='h',
            marker_color=SUGAR_COLOR[s], opacity=0.3,
        ))
    fig_top.add_trace(go.Bar(
        y=top['chimera_id'], x=top[col],
        name=sugar, orientation='h', marker_color=scolor,
    ))
    fig_top.update_layout(
        title=f'Top 20 Performers by {sugar} Activity',
        xaxis_title='Activity (ln(1+mRFU/s))',
        barmode='group', height=700,
        yaxis=dict(categoryorder='array', categoryarray=ranked_ids),
    )

    # --- 3. Activity Scatter: selected sugar vs each other sugar ---
    other_col = SUGAR_TO_COL[other_sugars[0]]
    size_col = SUGAR_TO_COL[other_sugars[1]]
    fig_scat = go.Figure()
    for cb, color in CREATED_BY_COLORS.items():
        sub = db_df[db_df['created_by'] == cb]
        if sub.empty:
            continue
        hover = sub.apply(lambda r, _s=sugar, _o0=other_sugars[0],
                          _o1=other_sugars[1], _col=col, _oc=other_col,
                          _sc=size_col: (
            f"ID: {r['chimera_id']}<br>"
            f"Creator: {r['created_by']}<br>"
            f"{_s}: {r[_col]:.2f}<br>"
            f"{_o0}: {r[_oc]:.2f}<br>"
            f"{_o1}: {r[_sc]:.2f}" +
            (f"<br>ZS Pred: {r['zero_shot_fitness_predictions']:.2f}"
             if pd.notna(r['zero_shot_fitness_predictions']) else '')
        ), axis=1)
        fig_scat.add_trace(go.Scatter(
            x=sub[col], y=sub[other_col],
            mode='markers', name=cb,
            marker=dict(
                size=sub[size_col].clip(lower=0) * 3 + 5,
                color=color, opacity=0.7,
                line=dict(width=1, color='black'),
            ),
            text=hover, hoverinfo='text',
        ))
    fig_scat.update_layout(
        title=f'Activity Scatter: {sugar} vs {other_sugars[0]} (size = {other_sugars[1]})',
        xaxis_title=f'{sugar} Activity',
        yaxis_title=f'{other_sugars[0]} Activity',
        height=600,
    )

    return fig_dist, fig_top, fig_scat


@app.callback(
    [Output('timecourse-graph', 'figure'),
     Output('activity-bar-graph', 'figure'),
     Output('metadata-panel', 'children')],
    [Input('enzyme-table', 'selected_rows'),
     Input('enzyme-table', 'data'),
     Input('plot-mode', 'value'),
     Input('median-toggle', 'value')],
)
def update_run_detail(selected_rows, table_data, plot_mode, median_toggle):
    empty = go.Figure()
    if not selected_rows or not table_data:
        return empty, empty, ''
    seq_id = table_data[selected_rows[0]]['sequence_id']

    entry = db_df[db_df['sequence_id'] == seq_id].iloc[0]
    chimera_id = entry['chimera_id']
    folder = chimera_to_folder.get(chimera_id)
    row_letter = chimera_to_row.get((chimera_id, folder)) if folder else None

    show_median = 'on' in (median_toggle or [])
    adjusted = plot_mode == 'adjusted'
    y_label = 'Adj RFU' if adjusted else 'RFU'
    slope_label = 'Adj Slope (RFU/min)' if adjusted else 'Slope (RFU/min)'

    # ------------------------------------------------------------------
    # 1. Timecourse (3 subplots: Glucose / Xylose / Mannose)
    # ------------------------------------------------------------------
    has_tc = False
    if folder and row_letter:
        try:
            fluor, tc = load_timecourse(folder)
            has_tc = not tc.empty and 'time_s' in tc.columns
        except Exception:
            has_tc = False

    if has_tc:
        # Apply normalization if adjusted
        norm_factors = {}
        if adjusted and not fluor.empty:
            norm_factors = compute_norm_factors(fluor.to_dict())
            for well in tc.columns:
                if well == 'time_s':
                    continue
                factor = norm_factors.get(well, 1.0)
                if factor != 0:
                    tc[well] = tc[well] / factor

        time_min = tc['time_s'] / 60.0
        time_s = tc['time_s'].values
        fig_tc = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Glucose', 'Xylose', 'Mannose'],
            shared_yaxes=False,
        )
        shown = set()

        rep_opacity = 0.25 if show_median else 1.0
        med_visible = show_median

        # Also collect slope data for the activity strip plot
        chimera_slopes = {}  # sugar -> [slope_rfu_per_min, ...]
        control_slopes = {}  # sugar -> [slope_rfu_per_min, ...]

        for si, (sugar, col_nums) in enumerate(SUGAR_COLS.items()):
            sugar_color = SUGAR_COLOR[sugar]
            all_y = []
            chimera_y_cols = []
            control_y_cols = []
            chimera_slopes[sugar] = []
            control_slopes[sugar] = []

            # Chimera wells
            for cn in col_nums:
                well = f'{row_letter}{cn}'
                if well not in tc.columns:
                    continue
                well_y = tc[well]
                all_y.extend(well_y.dropna().tolist())
                chimera_y_cols.append(well_y.values)
                sl = chimera_id not in shown
                if sl:
                    shown.add(chimera_id)
                fig_tc.add_trace(go.Scatter(
                    x=time_min, y=well_y, mode='lines',
                    name=chimera_id, legendgroup=chimera_id,
                    showlegend=sl,
                    opacity=rep_opacity,
                    line=dict(color=sugar_color),
                    hovertemplate=(f'{well} ({chimera_id})<br>'
                                  '%{x:.0f} min<br>%{y:.0f} ' + y_label +
                                  '<extra></extra>'),
                ), row=1, col=si + 1)
                # Compute slope
                valid = well_y.dropna()
                valid_t = time_s[:len(valid)]
                if len(valid) >= 2:
                    slope = stats.linregress(valid_t, valid.values).slope
                    chimera_slopes[sugar].append(slope * 60)

            # Chimera median trace
            if chimera_y_cols:
                stack = np.column_stack(chimera_y_cols)
                median_y = np.nanmedian(stack, axis=1)
                all_y.extend(median_y[~np.isnan(median_y)].tolist())
                fig_tc.add_trace(go.Scatter(
                    x=time_min, y=median_y, mode='lines',
                    name=f'{chimera_id} median', legendgroup=chimera_id,
                    showlegend=False,
                    visible=True if med_visible else 'legendonly',
                    opacity=1.0 if med_visible else 0,
                    line=dict(color=sugar_color, width=3),
                    hovertemplate=(f'{chimera_id} median<br>'
                                  '%{x:.0f} min<br>%{y:.0f} ' + y_label +
                                  '<extra></extra>'),
                ), row=1, col=si + 1)

            # Control wells (row G)
            for cn in col_nums:
                well = f'G{cn}'
                if well not in tc.columns:
                    continue
                well_y = tc[well]
                all_y.extend(well_y.dropna().tolist())
                control_y_cols.append(well_y.values)
                sl = 'Control' not in shown
                if sl:
                    shown.add('Control')
                fig_tc.add_trace(go.Scatter(
                    x=time_min, y=well_y, mode='lines',
                    name='Control', legendgroup='Control',
                    showlegend=sl,
                    opacity=rep_opacity,
                    line=dict(color='gray', dash='dash'),
                    hovertemplate=(f'{well} (Control)<br>'
                                  '%{x:.0f} min<br>%{y:.0f} ' + y_label +
                                  '<extra></extra>'),
                ), row=1, col=si + 1)
                # Compute slope
                valid = well_y.dropna()
                valid_t = time_s[:len(valid)]
                if len(valid) >= 2:
                    slope = stats.linregress(valid_t, valid.values).slope
                    control_slopes[sugar].append(slope * 60)

            # Control median trace
            if control_y_cols:
                stack = np.column_stack(control_y_cols)
                median_y = np.nanmedian(stack, axis=1)
                all_y.extend(median_y[~np.isnan(median_y)].tolist())
                fig_tc.add_trace(go.Scatter(
                    x=time_min, y=median_y, mode='lines',
                    name='Control median', legendgroup='Control',
                    showlegend=False,
                    visible=True if med_visible else 'legendonly',
                    opacity=1.0 if med_visible else 0,
                    line=dict(color='gray', width=3, dash='dash'),
                    hovertemplate=('Control median<br>'
                                  '%{x:.0f} min<br>%{y:.0f} ' + y_label +
                                  '<extra></extra>'),
                ), row=1, col=si + 1)

            # Set per-subplot y range with 25% padding
            if all_y:
                ymin, ymax = min(all_y), max(all_y)
                span = ymax - ymin if ymax != ymin else abs(ymax) or 1
                pad = span * 0.25
                axis_key = 'yaxis' if si == 0 else f'yaxis{si + 1}'
                fig_tc.update_layout(
                    **{axis_key: dict(range=[ymin - pad, ymax + pad])})

        fig_tc.update_layout(
            title=f'Timecourse \u2014 {chimera_id}',
            height=450,
        )
        fig_tc.update_xaxes(title_text='Time (min)')
        fig_tc.update_yaxes(title_text=y_label, col=1)

        # --------------------------------------------------------------
        # 2. Activity strip / dot plot from timecourse slopes
        #    (1x3 subplots with independent y-axes, matching timecourse)
        # --------------------------------------------------------------
        sugars_list = ['Glucose', 'Xylose', 'Mannose']
        fig_bar = make_subplots(
            rows=1, cols=3,
            subplot_titles=sugars_list,
            shared_yaxes=False,
        )
        dot_shown = set()

        for si, sugar in enumerate(sugars_list):
            sugar_color = SUGAR_COLOR[sugar]
            c_slopes = chimera_slopes.get(sugar, [])
            g_slopes = control_slopes.get(sugar, [])
            all_slopes = []

            # Chimera replicate dots
            if c_slopes:
                all_slopes.extend(c_slopes)
                jitter = np.linspace(-0.12, 0.12, len(c_slopes))
                sl = chimera_id not in dot_shown
                if sl:
                    dot_shown.add(chimera_id)
                if show_median:
                    med_val = float(np.median(c_slopes))
                    med_idx = int(np.argmin(np.abs(np.array(c_slopes) - med_val)))
                    opacities = [0.25 if i != med_idx else 1.0 for i in range(len(c_slopes))]
                else:
                    opacities = [1.0] * len(c_slopes)
                fig_bar.add_trace(go.Scatter(
                    x=[0 + j for j in jitter], y=c_slopes,
                    mode='markers',
                    name=chimera_id, legendgroup=chimera_id,
                    showlegend=sl,
                    marker=dict(color=sugar_color, size=9,
                                opacity=opacities,
                                line=dict(width=1, color='black')),
                    hovertemplate=(f'{chimera_id}<br>{sugar}<br>'
                                  '%{y:.2f} ' + slope_label +
                                  '<extra></extra>'),
                ), row=1, col=si + 1)

            # Control replicate dots
            if g_slopes:
                all_slopes.extend(g_slopes)
                jitter = np.linspace(-0.12, 0.12, len(g_slopes))
                sl = 'Control' not in dot_shown
                if sl:
                    dot_shown.add('Control')
                if show_median:
                    med_val = float(np.median(g_slopes))
                    med_idx = int(np.argmin(np.abs(np.array(g_slopes) - med_val)))
                    opacities = [0.25 if i != med_idx else 1.0 for i in range(len(g_slopes))]
                else:
                    opacities = [1.0] * len(g_slopes)
                fig_bar.add_trace(go.Scatter(
                    x=[1 + j for j in jitter], y=g_slopes,
                    mode='markers',
                    name='Control', legendgroup='Control',
                    showlegend=sl,
                    marker=dict(color='gray', size=9,
                                opacity=opacities,
                                line=dict(width=1, color='black')),
                    hovertemplate=(f'Control<br>{sugar}<br>'
                                  '%{y:.2f} ' + slope_label +
                                  '<extra></extra>'),
                ), row=1, col=si + 1)

            # Per-subplot y range with 25% padding (same as timecourse)
            if all_slopes:
                ymin, ymax = min(all_slopes), max(all_slopes)
                span = ymax - ymin if ymax != ymin else abs(ymax) or 1
                pad = span * 0.25
                axis_key = 'yaxis' if si == 0 else f'yaxis{si + 1}'
                fig_bar.update_layout(
                    **{axis_key: dict(range=[ymin - pad, ymax + pad])})

            # X-axis: categorical ticks for chimera vs control
            xaxis_key = 'xaxis' if si == 0 else f'xaxis{si + 1}'
            fig_bar.update_layout(
                **{xaxis_key: dict(
                    tickvals=[0, 1],
                    ticktext=[chimera_id, 'Control'],
                )})

        fig_bar.update_layout(
            title=f'Activity \u2014 {chimera_id}',
            height=400,
        )
        fig_bar.update_yaxes(title_text=slope_label, col=1)
    else:
        fig_tc = go.Figure().update_layout(
            title='No timecourse data available')

        # Fallback: DB activity values as simple bar chart
        sugars_list = ['Glucose', 'Xylose', 'Mannose']
        values = [entry['a1'], entry['a2'], entry['a3']]
        colors_bar = ['#5a63a6', '#ff4470', '#ffa600']
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=sugars_list, y=values,
                                 marker_color=colors_bar))
        fig_bar.update_layout(
            title=f'Activity \u2014 {chimera_id} (from database)',
            yaxis_title='Activity (ln(1+mRFU/s))',
            height=350,
        )

    # ------------------------------------------------------------------
    # 3. Metadata panel
    # ------------------------------------------------------------------
    zs = entry['zero_shot_fitness_predictions']
    zs_str = f'{zs:.3f}' if pd.notna(zs) else 'N/A'
    metadata = html.Div([
        html.H3(f'Enzyme: {chimera_id}'),
        html.Table([
            html.Tr([
                html.Td('Chimera ID:',
                         style={'fontWeight': 'bold', 'paddingRight': '15px'}),
                html.Td(chimera_id)]),
            html.Tr([
                html.Td('Created By:',
                         style={'fontWeight': 'bold', 'paddingRight': '15px'}),
                html.Td(entry['created_by'])]),
            html.Tr([
                html.Td('Created At:',
                         style={'fontWeight': 'bold', 'paddingRight': '15px'}),
                html.Td(str(entry['created_at']))]),
            html.Tr([
                html.Td('ZS Fitness Prediction:',
                         style={'fontWeight': 'bold', 'paddingRight': '15px'}),
                html.Td(zs_str)]),
            html.Tr([
                html.Td('Archive Folder:',
                         style={'fontWeight': 'bold', 'paddingRight': '15px'}),
                html.Td(folder or 'N/A')]),
        ], style={'fontSize': '14px'}),
    ], style={'padding': '10px', 'backgroundColor': '#f8f9fa',
              'borderRadius': '5px', 'marginTop': '10px'})

    return fig_tc, fig_bar, metadata


if __name__ == '__main__':
    app.run(debug=True, port=8050)
