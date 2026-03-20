import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project = 'dual_mpcritic_ddpg_new_value_with_discount'

runs_df = pd.read_pickle(f"data/{project}_all_data.pkl")
print(runs_df['episodic_return'].iloc[0])

data_env = runs_df[(runs_df['env_id']=='Hopper-v5')]
data_target_r20 = data_env[(data_env['num_target_rollouts']==20)]
data_target_h4 = data_target_r20[(data_target_r20['target_horizon']==4)]
data = pd.concat([data_target_h4], ignore_index=True)
# data = pd.concat([data_target_h4.sample(n=4)], ignore_index=True)

data['label1'] = ""
# subplot1: MPPI Parameterization (model & Q in MPPI) trends
condition_0 = (data['mppi_online'] == True)
condition_1 = (pd.isna(data['transition_ensemble_size']))
condition_2 = (data['Q_in_mppi'])
condition_3 = (data['mppi_targets'])
data.loc[condition_0 & ~condition_1 & condition_2, 'label1'] = r'$f$ Ensemble'
data.loc[condition_0 & condition_2 & ~condition_1, 'label1'] = r'$Q$'
data.loc[condition_0 & condition_1 & condition_2 & condition_3, 'label1'] = r'$f$ Ensemble $+$ $Q$'
data['label1'] = data['label1'].astype("category")

data['label2'] = ""
# subplot2: MPPI Usage in RL trends
condition_0 = (pd.isna(data['transition_ensemble_size'])) & (data['Q_in_mppi'])
condition_1 = (data['mppi_targets'])
condition_2 = (data['mppi_online'])
data.loc[condition_0 & condition_1 & ~condition_2, 'label2'] = 'Targets'
data.loc[condition_0 & condition_2 & ~condition_1, 'label2'] = 'Control'
data.loc[condition_0 & condition_1 & condition_2, 'label2'] = 'Targets $+$ Control'
data['label2'] = data['label2'].astype("category")

def padded_moving_average(df, columns, window=20, total_len=500_001):
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

                df.at[i, 'global_step'] = repeated_steps[::window]
                df.at[i, column] = repeated_values[::window]
    return df

data = padded_moving_average(data, ['episodic_return'])

df_long = data.explode(['episodic_return', 'global_step'], ignore_index=True)

hue1 = 'label1'
hue2 = 'label2'

sns.set(palette='Set2', style='ticks')

SMALL_SIZE = 8 # 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    #  fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
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
    "estimator" : np.median,
    "errorbar" : min_max_error,
    "alpha" : 1.0
}
subplot1_kwargs = {**lineplot_kwargs, **{
    "hue" : hue1,
    "style" : hue1,
}}
subplot2_kwargs = {**lineplot_kwargs, **{
    "hue" : hue2,
    "style" : hue2,
}}

# fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(3.3,3.3), layout='constrained')
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(3.5,2.5))
plt.tight_layout()

# Plot on the first axis
# filtered1 = df_long[df_long['label1'] != ''].copy()
# filtered1['label1'] = filtered1['label1'].astype('category')
# sns.lineplot(data=filtered1, ax=axes[0], **subplot1_kwargs)
g1 = sns.lineplot(data=df_long[df_long['label1'] != ''], ax=axes[0], **subplot1_kwargs)
g1.set_xlabel("Time step")
g1.set_ylabel("Reward")
axes[0].legend()
sns.move_legend(axes[0], "lower center", title="MPPI Ingredients", bbox_to_anchor=(0.5, 1), frameon=False, ncol=1)

# Plot on the second axis
# filtered2 = df_long[df_long['label2'] != ''].copy()
# filtered2['label2'] = filtered2['label2'].astype('category')
# sns.lineplot(data=filtered2, ax=axes[1], **subplot2_kwargs,)
g2 = sns.lineplot(data=df_long[df_long['label2'] != ''].copy(), ax=axes[1], **subplot2_kwargs,)
g2.set_xlabel("Time step")
axes[1].legend()
sns.move_legend(axes[1], "lower center", title="MPPI Usage", bbox_to_anchor=(0.5, 1), frameon=False, ncol=1)

# g.set_axis_labels("Time step", "Cumulative reward")
# g.set_titles(col_template="{col_name} start targets", row_template="{row_name} dynamics model")
# g.tight_layout() # call before making legend outside
# g.add_legend()
# sns.move_legend(g, "lower center", title="MPPI target iterations", bbox_to_anchor=(0.5, 0.98), frameon=False, ncol=3)

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles, labels=['Vanilla', 'MPCritic'], title=None)
# sns.move_legend(ax, "lower center", title=None, bbox_to_anchor=(0.5, 1), ncol=2)

# plt.tick_params(axis='both', which='major', top=True, right=True, bottom=True, left=True, length=5, width=1)

# axes[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
# axes[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
plt.savefig('plotting/hop_eval.png', bbox_inches='tight')
plt.savefig('plotting/hop_eval.pdf', bbox_inches='tight')