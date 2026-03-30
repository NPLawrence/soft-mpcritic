import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project = 'sac_baseline'
runs_df_sac = pd.read_pickle(f"data/{project}_all_data.pkl")
runs_df_sac = runs_df_sac[(runs_df_sac['env_id']=='Hopper-v5')]
runs_df_sac = pd.concat([runs_df_sac], ignore_index=True)
runs_df_sac['label'] = "SAC"
runs_df_sac['label'] = runs_df_sac['label'].astype("category")

project = 'ddpg_baseline'
runs_df_ddpg = pd.read_pickle(f"data/{project}_all_data.pkl")
runs_df_ddpg = runs_df_ddpg[(runs_df_ddpg['env_id']=='Hopper-v5')]
runs_df_ddpg = pd.concat([runs_df_ddpg], ignore_index=True)
runs_df_ddpg['label'] = "DDPG"
runs_df_ddpg['label'] = runs_df_ddpg['label'].astype("category")

project = 'dual_mpcritic_ddpg_hopper'
runs_df_mppi = pd.read_pickle(f"data/{project}_all_data.pkl")
runs_df_mppi = runs_df_mppi[(runs_df_mppi['env_id']=='Hopper-v5')]
runs_df_mppi = runs_df_mppi[(runs_df_mppi['training_pattern']=='online')]
runs_df_mppi = runs_df_mppi[(runs_df_mppi['target_horizon']==4)]
runs_df_mppi = pd.concat([runs_df_mppi], ignore_index=True)
runs_df_mppi['label'] = r"\texttt{soft\,MPCritic}"
runs_df_mppi['label'] = runs_df_mppi['label'].astype("category")

print(runs_df_mppi['episodic_return'].iloc[0])

def padded_moving_average(df, columns, window=20, total_len=500_001, final_len=500_001):
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

                df.at[i, 'global_step'] = repeated_steps[:final_len][::window]
                df.at[i, column] = repeated_values[:final_len][::window]
    return df

plotted_columns = ['global_step', 'episodic_return', 'label']
sac_data = runs_df_sac[plotted_columns]
sac_data = padded_moving_average(runs_df_sac, ['episodic_return'], window=20, total_len=1_000_001)
sac_df_long = sac_data.explode(['episodic_return', 'global_step'], ignore_index=True)

ddpg_data = runs_df_ddpg[plotted_columns]
ddpg_data = padded_moving_average(runs_df_ddpg, ['episodic_return'], window=20, total_len=1_000_001)
ddpg_df_long = ddpg_data.explode(['episodic_return', 'global_step'], ignore_index=True)

mppi_data = runs_df_mppi[plotted_columns]
mppi_data = padded_moving_average(runs_df_mppi, ['episodic_return'], window=20)
mppi_df_long = mppi_data.explode(['episodic_return', 'global_step'], ignore_index=True)

df_long = pd.concat([sac_df_long, ddpg_df_long, mppi_df_long])
hue = 'label'

sns.set(palette='husl', style='ticks')

SMALL_SIZE = 8 # 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    #  fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('xtick.major', size=5, width=1)    #  fontsize of the tick labels
plt.rc('ytick.major', size=5, width=1)    # fontsize of the tick labels
plt.rc('legend', title_fontsize=SMALL_SIZE, fontsize=SMALL_SIZE)    # legend fontsize
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
    "estimator" : "median", # np.median
    "errorbar" : ("pi", 80), # min_max_error
    "alpha" : 1.0,
}
# r'$f$ Ensemble', r'$Q$', r'$f$ Ensemble $+$ $Q$'
plot_kwargs = {**lineplot_kwargs, **{
    "hue" : hue,
    "style" : hue,
    "palette": {"SAC": sns.color_palette()[0],
                "DDPG": sns.color_palette()[1],
                r"\texttt{soft\,MPCritic}": sns.color_palette()[2]},
    "dashes": {"SAC": (3,1),
               "DDPG": (1,4.5),
               r"\texttt{soft\,MPCritic}": (1,0),
               '': (0,0)},
}}

fig, ax = plt.subplots(figsize=(3.3, 3.3))
plt.tight_layout()

# Plot on axis
ax = sns.lineplot(data=df_long, ax=ax, **plot_kwargs)
ax.set_xlabel("Time step")
ax.set_ylabel("Cumulative reward")
ax.grid(True)
ax.legend()
sns.move_legend(ax, "lower center", title="", bbox_to_anchor=(0.5, 1), frameon=False, ncol=3)

# fine-tuning x-ticks
xticks = np.arange(0, 5*10e4+1, step=10e4)
yticks = np.arange(0, 3*10e2+1, step=5*10e1)
ax.set(xticks=xticks, yticks=yticks)
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
# ax.xaxis.offsetText.set_visible(False)

plt.tight_layout(h_pad=0.5, w_pad=0.5) # need this to squeeze plots together

plt.savefig('plotting/hop_method.png', bbox_inches='tight')
plt.savefig('plotting/hop_method.pdf', bbox_inches='tight')