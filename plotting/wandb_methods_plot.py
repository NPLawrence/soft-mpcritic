import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

WINDOW = 5000
TOTAL_LEN = 500_000
FINAL_LEN = 500_000

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

project = 'dual_mpcritic_ddpg_hopper_tb'
runs_df_mppi = pd.read_pickle(f"data/{project}_all_data.pkl")
runs_df_mppi = runs_df_mppi[(runs_df_mppi['env_id']=='Hopper-v5')]
runs_df_mppi = runs_df_mppi[(runs_df_mppi['training_pattern']=='online')]
runs_df_mppi = runs_df_mppi[(runs_df_mppi['target_horizon']==4)]
runs_df_mppi = runs_df_mppi[(runs_df_mppi['num_rollouts']==600)]
runs_df_mppi = runs_df_mppi[(runs_df_mppi['lambda_']==0.15)]
runs_df_mppi = runs_df_mppi[(runs_df_mppi['var']==0.05)]
runs_df_mppi = pd.concat([runs_df_mppi], ignore_index=True)
runs_df_mppi['label'] = r"\texttt{soft\,MPCritic}"
runs_df_mppi['label'] = runs_df_mppi['label'].astype("category")

print(runs_df_mppi['episodic_return'].iloc[0])

def pad_arrays(df, column, total_len=500_000,):
    for i in range(len(df)):
        if df.at[i, 'global_step'] is not None:
            diffs = np.diff(df.at[i, 'global_step'])
            repeated_values = np.repeat(df.at[i, column][:-1], diffs)
            pad_width = total_len - len(repeated_values)
            repeated_values = np.pad(repeated_values, (0, pad_width), mode='edge')

            repeated_steps  = np.arange(total_len)

            df.at[i, 'global_step'] = repeated_steps
            df.at[i, column] = repeated_values

    return df

def median_low_high(df, column):
    run_arrays = []
    for i in range(len(df)):
        run_arrays.append(df.at[i, column][None,:])

    run_arrays = np.concatenate(run_arrays)
    run_medians = np.median(run_arrays, axis=0)
    run_lows = np.percentile(run_arrays, q=20, axis=0, interpolation='midpoint')
    run_highs = np.percentile(run_arrays, q=80, axis=0, interpolation='midpoint')

    return run_medians, run_lows, run_highs

def moving_average(array):
    ma = pd.Series(array).rolling(WINDOW).mean().dropna().to_numpy()
    return ma

def trim(array):
    return array[:FINAL_LEN][::WINDOW]

def ma_global_step():
    return np.arange(WINDOW, FINAL_LEN+1, WINDOW)

runs_df_mppi = pad_arrays(runs_df_mppi, 'episodic_return')
med_mppi, low_mppi, high_mppi = median_low_high(runs_df_mppi, 'episodic_return')
med_mppi, low_mppi, high_mppi = map(moving_average, [med_mppi, low_mppi, high_mppi])
med_mppi, low_mppi, high_mppi = map(trim, [med_mppi, low_mppi, high_mppi])

runs_df_ddpg = pad_arrays(runs_df_ddpg, 'episodic_return', total_len=1_000_000)
med_ddpg, low_ddpg, high_ddpg = median_low_high(runs_df_ddpg, 'episodic_return')
med_ddpg, low_ddpg, high_ddpg = map(moving_average, [med_ddpg, low_ddpg, high_ddpg])
ddpg_median_final_value = med_ddpg[-1]
med_ddpg, low_ddpg, high_ddpg = map(trim, [med_ddpg, low_ddpg, high_ddpg])

runs_df_sac = pad_arrays(runs_df_sac, 'episodic_return', total_len=1_000_000)
med_sac, low_sac, high_sac = median_low_high(runs_df_sac, 'episodic_return')
med_sac, low_sac, high_sac = map(moving_average, [med_sac, low_sac, high_sac])
sac_median_final_value = med_sac[-1]
med_sac, low_sac, high_sac = map(trim, [med_sac, low_sac, high_sac])

mppi_df = pd.DataFrame({
    'global_step' : ma_global_step(),
    'median': med_mppi,
    'low': low_mppi,
    'high': high_mppi,
    'label': r"\texttt{soft\,MPCritic}"
})

ddpg_df = pd.DataFrame({
    'global_step' : ma_global_step(),
    'median': med_ddpg,
    'low': low_ddpg,
    'high': high_ddpg,
    'label': "DDPG"
})

sac_df = pd.DataFrame({
    'global_step' : ma_global_step(),
    'median': med_sac,
    'low': low_sac,
    'high': high_sac,
    'label': "SAC"
})

plot_df = pd.concat([mppi_df, ddpg_df, sac_df], ignore_index=True)
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

fig, ax = plt.subplots(figsize=(3.5,3.0))
plt.tight_layout()

# lineplot_kwargs = {
#     "x" : "global_step",
#     "y" : "episodic_return",
#     "estimator" : np.median, # "median", # np.median
#     "errorbar" : min_max_error, # ("pi", 80), # min_max_error
#     "alpha" : 1.0,
# }
# r'$f$ Ensemble', r'$Q$', r'$f$ Ensemble $+$ $Q$'
plot_kwargs = {
    "hue" : hue,
    "style" : hue,
    "palette": {"SAC": sns.color_palette()[0],
                "DDPG": sns.color_palette()[1],
                r"\texttt{soft\,MPCritic}": sns.color_palette()[4]},
    "dashes": {"SAC": (3,1),
               "DDPG": (1,4.5),
               r"\texttt{soft\,MPCritic}": (1,0),
               '': (0,0)},
}

ax.axhline(y=sac_median_final_value, color=plot_kwargs["palette"]["SAC"], dashes=plot_kwargs["dashes"]["SAC"], label='')
ax.axhline(y=ddpg_median_final_value, color=plot_kwargs["palette"]["DDPG"], dashes=plot_kwargs["dashes"]["DDPG"], label='')
# ax.axhline(y=sac_median_final_value, color="grey", dashes=plot_kwargs["dashes"]["SAC"], lw=1.5, label='')
# ax.axhline(y=ddpg_median_final_value, color="grey", dashes=plot_kwargs["dashes"]["DDPG"], lw=1.5, label='')
ax = sns.lineplot(plot_df, ax=ax, x='global_step', y='median', **plot_kwargs)

palette_key = list(plot_kwargs["palette"].keys())[0]
color = plot_kwargs["palette"][palette_key]
ax.fill_between(x=sac_df['global_step'], y1=sac_df['low'], y2=sac_df['high'], color=color, alpha=0.2)

palette_key = list(plot_kwargs["palette"].keys())[1]
color = plot_kwargs["palette"][palette_key]
ax.fill_between(x=ddpg_df['global_step'], y1=ddpg_df['low'], y2=ddpg_df['high'], color=color, alpha=0.2)

palette_key = list(plot_kwargs["palette"].keys())[2]
color = plot_kwargs["palette"][palette_key]
ax.fill_between(x=mppi_df['global_step'], y1=mppi_df['low'], y2=mppi_df['high'], color=color, alpha=0.2)

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

plt.tight_layout(h_pad=0.5, w_pad=0.5) # need this to squeeze plots together

plt.savefig('plotting/hop_method.png', bbox_inches='tight')
plt.savefig('plotting/hop_method.pdf', bbox_inches='tight')