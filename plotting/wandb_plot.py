import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project = 'dual_mpcritic_ddpg_new_value_with_discount'

runs_df = pd.read_pickle(f"data/{project}_all_data.pkl")
print(runs_df['episodic_return'].iloc[0])

data_env = runs_df[(runs_df['env_id']=='InvertedDoublePendulum-v5')]
data_target_r20 = data_env[(data_env['num_target_rollouts']==20)]
data_target_h1 = data_target_r20[(data_target_r20['target_horizon']==1)]
data_target_h4 = data_target_r20[(data_target_r20['target_horizon']==4)]
data = pd.concat([data_target_h4], ignore_index=True)
# data = pd.concat([data_target_h4.sample(n=20)], ignore_index=True)

def padded_moving_average(df, columns, window=20, total_len=50_001):
    for column in columns:
        for i in range(len(df)):
            df.at[i, column] = pd.Series(df[column].iloc[i]).rolling(window).mean().dropna().to_numpy()
            if df['global_step'] is not None:
                df.at[i, 'global_step'] = df.at[i, 'global_step'][window-1:]

                diffs = np.diff(df.at[i, 'global_step'])
                repeated_values = np.repeat(df.at[i, column][:-1], diffs)
                pad_width = total_len - len(repeated_values)
                repeated_values = np.pad(repeated_values, (0, pad_width), mode='edge')

                repeated_steps  = np.arange(total_len)

                df.at[i, 'global_step'] = repeated_steps
                df.at[i, column] = repeated_values
    return df

data = padded_moving_average(data, ['episodic_return'])

df_long = data.explode(['episodic_return', 'global_step'], ignore_index=True)
df_long.loc[df_long['mppi_target_warmstart'] == True, 'mppi_target_iterations'] = '1 (Warm)'
df_long.loc[df_long['mppi_target_warmstart'] == False, 'mppi_target_warmstart'] = 'Cold'
df_long.loc[df_long['mppi_target_warmstart'] == True, 'mppi_target_warmstart'] = 'Warm'
df_long.loc[~pd.isna(df_long['transition_ensemble_size']), 'transition_ensemble_size'] = 'Single'
df_long.loc[pd.isna(df_long['transition_ensemble_size']), 'transition_ensemble_size'] = 'Ensemble'
hue = 'mppi_target_iterations'

sns.set(palette='Set2', style='ticks')

SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    #  fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
plt.rc('lines', linewidth=2.5)

params = {
        "text.usetex" : True,
        "font.family" : "serif",
        "font.serif" : ["Computer Modern Serif"]}
plt.rcParams.update(params)

def min_max_error(x):
    return x.min(), x.max()

lineplot_kwargs = {
    "x" : "global_step",
    "y" : "episodic_return",
    "hue" : hue,
    "estimator" : np.median,
    "errorbar" : min_max_error,
    "style" : hue,
    "alpha" : 1.0
}

# compute a moving median/average (and percentiles/std of that) over seeds of shared parameters???

g = sns.FacetGrid(df_long, col="mppi_target_warmstart", row="transition_ensemble_size", margin_titles=True, despine=False, legend_out=False)
# g.fig.set_constrained_layout(True)
g.map_dataframe(sns.lineplot, **lineplot_kwargs)
g.set_axis_labels("Time step", "Cumulative reward")
g.set_titles(col_template="{col_name} start targets", row_template="{row_name} dynamics model")
g.tight_layout() # call before making legend outside
g.add_legend()
sns.move_legend(g, "lower center", title="MPPI target iterations", bbox_to_anchor=(0.5, 0.98), frameon=False, ncol=3)

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles, labels=['Vanilla', 'MPCritic'], title=None)
# sns.move_legend(ax, "lower center", title=None, bbox_to_anchor=(0.5, 1), ncol=2)

# plt.tick_params(axis='both', which='major', top=True, right=True, bottom=True, left=True, length=5, width=1)

plt.savefig('plotting/dip_eval.png', bbox_inches='tight')
plt.savefig('plotting/dip_eval.pdf', bbox_inches='tight')