#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_regret_data():
    """Load the cumulative regret data from SQL source"""
    try:
        df = pd.read_csv('sql_cumulative_regret_pivot.csv')
        return df
    except FileNotFoundError:
        print("Error: sql_cumulative_regret_pivot.csv not found!")
        print("Please run plot_sql_cumulative_regret.py first to generate SQL-based regret data")
        return None

def calculate_regret_per_round(cumulative_regret):
    """Calculate regret added per round from cumulative regret"""
    regret_per_round = [cumulative_regret[0]]  # First round regret
    for i in range(1, len(cumulative_regret)):
        regret_per_round.append(cumulative_regret[i] - cumulative_regret[i-1])
    return regret_per_round

def analyze_regret_trend(rounds, cumulative_regret, substrate_name):
    """Analyze the regret trend for a substrate (silent)"""
    # Calculate regret per round
    regret_per_round = calculate_regret_per_round(cumulative_regret)
    
    # Fit to logarithmic curve
    log_rounds = [np.log(r) for r in rounds]
    log_slope, log_intercept, log_r_value, log_p_value, log_std_err = stats.linregress(log_rounds, cumulative_regret)
    
    # Fit to linear curve
    linear_slope, linear_intercept, linear_r_value, linear_p_value, linear_std_err = stats.linregress(rounds, cumulative_regret)
    
    # Calculate fitted regression lines
    fitted_linear_regret = [linear_slope * r + linear_intercept for r in rounds]
    fitted_log_regret = [log_slope * np.log(r) + log_intercept for r in rounds]
    
    # Phase assessment (silent)
    if linear_r_value**2 > 0.9 and linear_r_value**2 > log_r_value**2:
        phase_assessment = "Strong exploration phase (linear growth)"
    elif log_r_value**2 > 0.9 and log_r_value**2 > linear_r_value**2:
        phase_assessment = "Strong exploitation phase (log growth)"
    else:
        phase_assessment = "Transitional phase"
    
    return {
        'regret_per_round': regret_per_round,
        'fitted_log_regret': fitted_log_regret,
        'fitted_linear_regret': fitted_linear_regret,
        'correlation': stats.pearsonr(rounds, regret_per_round)[0],
        'log_r_squared': log_r_value**2,
        'linear_r_squared': linear_r_value**2,
        'linear_slope': linear_slope,
        'phase_assessment': phase_assessment,
        'early_avg_regret': np.mean(regret_per_round[:5]),
        'late_avg_regret': np.mean(regret_per_round[-5:])
    }

def plot_regret_analysis(df):
    """Create combined analysis plots: cumulative max with per-round underlay"""
    fig, axes = plt.subplots(1, 3, figsize=(30, 6))
    fig.suptitle('UCB Analysis: Cumulative Max with Per-Round Specificity (SQL)', fontsize=16)
    
    # Style each axis
    for j in range(3):
        axes[j].set_facecolor('#ffffff')
        for spine in axes[j].spines.values():
            spine.set_linewidth(3)
        # Longer ticks
        axes[j].tick_params(which='both', width=2, labelsize=12, length=8)
        # Disable minor ticks
        axes[j].minorticks_off()
    
    substrates = ['spec1', 'spec2', 'spec3']
    sub_names = ['glu', 'xyl', 'man']
    colors = ['#5a63a4', '#fd4470', '#fda404']
    
    # Load SQL round specificity data
    try:
        sql_round_df = pd.read_csv('sql_round_specificity_summary.csv')
    except FileNotFoundError:
        sql_round_df = None
    
    results = {}
    
    for i, (substrate, sub_name) in enumerate(zip(substrates, sub_names)):
        if substrate in df.columns:
            # Use all rounds with forward-filled cumulative regret for analysis (console only)
            series = df[substrate].ffill()
            rounds = df['round'].tolist()
            cumulative_regret = series.tolist()
            rounds_clean = rounds
            
            # Analyze (prints only)
            analysis = analyze_regret_trend(rounds_clean, cumulative_regret, substrate)
            results[substrate] = analysis
            
            ax = axes[i]
            
            if sql_round_df is not None:
                sql_rounds = sql_round_df['round'].tolist()
                sql_spec_col = f'best_{substrate}'
                if sql_spec_col in sql_round_df.columns:
                    sql_specificities = sql_round_df[sql_spec_col].tolist()
                    
                    # Compute cumulative max for top curve
                    cumulative_max = []
                    running_max = 0
                    for spec in sql_specificities:
                        running_max = max(running_max, spec)
                        cumulative_max.append(running_max)
                    
                    # Plot per-round (underlay, lower opacity)
                    ax.plot(sql_rounds, sql_specificities, '-', color=colors[i],
                            linewidth=3, alpha=0.25, label='Per-round best')
                    
                    # Overlay cumulative max (higher opacity/thicker)
                    ax.plot(sql_rounds, cumulative_max, '-', color=colors[i],
                            linewidth=5, alpha=0.9, label='Cumulative max')
                    
                    # Final max line
                    final_max = cumulative_max[-1]
                    ax.axhline(y=final_max, color=colors[i], linestyle='--', alpha=0.4, linewidth=3)
                    
                    ax.set_xlabel('Round')
                    ax.set_ylabel('Specificity')
                    ax.set_title(f'{sub_name.upper()}: Cumulative vs Per-Round')
                    # Whole-number ticks every 2 rounds
                    max_round = max(sql_rounds)
                    ax.set_xticks(range(2, max_round + 1, 2))
                    
                    # Vertical lines per round
                    for round_num in sql_rounds:
                        ax.axvline(x=round_num, color='lightgray', linestyle='-', alpha=0.25, linewidth=1.5)
                    
                    # Removed legend per request
    
    # Final styling for ticks
    for j in range(3):
        for label in axes[j].get_xticklabels() + axes[j].get_yticklabels():
            label.set_fontweight('normal')
            label.set_fontfamily('sans-serif')
    
    plt.tight_layout()
    plt.savefig('ucb_regret_analysis_combined.png', dpi=800, bbox_inches='tight')
    plt.show()
    
    return results

def main():
    """Main analysis function (silent except top-3 prints)"""
    df = load_regret_data()
    if df is None:
        return
    
    # Generate figure silently
    _ = plot_regret_analysis(df)
    
    # Print only top-3 specificities per substrate from round summary CSV
    try:
        sql_round_df = pd.read_csv('sql_round_specificity_summary.csv')
        top_spec1 = sql_round_df['best_spec1'].nlargest(3).round(3).tolist()
        top_spec2 = sql_round_df['best_spec2'].nlargest(3).round(3).tolist()
        top_spec3 = sql_round_df['best_spec3'].nlargest(3).round(3).tolist()
        print(f"GLU (spec1) top 3: {top_spec1}")
        print(f"XYL (spec2) top 3: {top_spec2}")
        print(f"MAN (spec3) top 3: {top_spec3}")
    except Exception:
        pass

if __name__ == '__main__':
    main() 