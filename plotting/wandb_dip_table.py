import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project = 'dual_mpcritic_ddpg_sps'

runs_df = pd.read_pickle(f"data/{project}_all_data.pkl")
data = pd.concat([runs_df], ignore_index=True)

def trim_series(df, column='sps'):
    for i in range(len(df)):
        midpoint = len(df[column].iloc[i]) // 2
        df.at[i, column] = df[column].iloc[i][midpoint:]
    return df

data = trim_series(data, 'sps')

df_long = data.explode(['sps'], ignore_index=True)
df_long.loc[df_long['mppi_target_warmstart'] == True, 'mppi_target_iterations'] = '1 (Warm)'
mean = df_long.groupby(['num_target_rollouts', 'mppi_target_iterations'])['sps'].mean().to_dict()
std = df_long.groupby(['num_target_rollouts', 'mppi_target_iterations'])['sps'].std().to_dict()

row_labels = pd.unique(df_long['num_target_rollouts'])
col_labels = pd.unique(df_long['mppi_target_iterations'])

def format_sci_latex(value, decimals=1):
    if value == 0:
        return f"$0$"
    exp = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10 ** exp)
    return f"${mantissa:.{decimals}f}{{\\times}}10^{{{exp}}}$"

def print_table(mean, std, row_labels, col_labels):
    for row_label in row_labels:
        row_parts = [str(row_label)]
        for col_label in col_labels:
            key = (row_label, col_label)
            m = mean.get(key, float('nan'))
            s = std.get(key, float('nan'))
            mean_str = f"{m:.0f}" # format_sci_latex(m)
            std_str = f"{s:.2f}" # format_sci_latex(s)
            row_parts.append(f"{mean_str}$\\pm${std_str}")
        print(" & ".join(row_parts) + " \\\\")

print_table(mean, std, row_labels, col_labels)