import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

WINDOW = 5000
TOTAL_LEN = 500_000
FINAL_LEN = 500_000

project = 'dual_mpcritic_ddpg_new_value_with_discount'

runs_df = pd.read_pickle(f"data/{project}_all_data.pkl")
print(runs_df['episodic_return'].iloc[0])

data_env = runs_df[(runs_df['env_id']=='Hopper-v5')]
data_target_r20 = data_env[(data_env['num_target_rollouts']==20) & (data_env['num_rollouts']==200)]
data_target_h4 = data_target_r20[(data_target_r20['target_horizon']==4)]
data = pd.concat([data_target_h4], ignore_index=True)

data['label1'] = ""
# subplot1: MPPI Parameterization (model & Q in MPPI) trends
condition_0 = (data['mppi_online'] == True)
condition_1 = (pd.isna(data['transition_ensemble_size']))
condition_2 = (data['Q_in_mppi'])
condition_3 = (data['mppi_targets'])
condition_4 = (data['horizon']==4)

labels1 = [
    r'$\mathrm{no}\ \mathcal{Q}\ /\ f\ \mathrm{Ensemble}$',
    r'$\mathcal{Q}\ /\ \mathrm{no}\ f\ \mathrm{Ensemble}$',
    r'$f\ \mathrm{Ensemble} + \mathcal{Q}$'
]
data.loc[condition_0 & condition_1 & ~condition_2 & ~condition_3 & condition_4, 'label1'] = labels1[0]
data.loc[condition_0 & ~condition_1 & condition_2, 'label1'] = labels1[1]
data.loc[condition_0 & condition_1 & condition_2 & condition_3, 'label1'] = labels1[2]
data['label1'] = data['label1'].astype("category")

data['label2'] = ""
# subplot2: MPPI Usage in RL trends
condition_0 = (pd.isna(data['transition_ensemble_size'])) & (data['Q_in_mppi'])
condition_1 = (data['mppi_targets'])
condition_2 = (data['mppi_online'])

labels2 = ['Targets', 'Control', 'Targets $+$ Control']
data.loc[condition_0 & condition_1 & ~condition_2, 'label2'] = labels2[0]
data.loc[condition_0 & ~condition_1 & condition_2, 'label2'] = labels2[1]
data.loc[condition_0 & condition_1 & condition_2, 'label2'] = labels2[2]
data['label2'] = data['label2'].astype("category")

def pad_arrays(df, column, total_len=500_000,):
    for i in df.index:
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
    for i in df.index:
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

def process_data(data, y_column, label_column, label, total_len=500_000):
    data = data[data[label_column] == label]
    data = pad_arrays(data, y_column, total_len)
    med, low, high = median_low_high(data, y_column)
    med, low, high = map(moving_average, [med, low, high])
    med, low, high = map(trim, [med, low, high])
    return pd.DataFrame({
        'global_step' : ma_global_step(),
        'median': med,
        'low': low,
        'high': high,
        'label': label
    })

no_Q_df = process_data(data, 'episodic_return', 'label1', labels1[0])
no_f_ens_df = process_data(data, 'episodic_return', 'label1', labels1[1])
Q_and_ens_df = process_data(data, 'episodic_return', 'label1', labels1[2])

target_df = process_data(data, 'episodic_return', 'label2', labels2[0])
control_df = process_data(data, 'episodic_return', 'label2', labels2[1])
target_and_control_df = process_data(data, 'episodic_return', 'label2', labels2[2])

plot1_df = pd.concat([no_Q_df, no_f_ens_df, Q_and_ens_df], ignore_index=True)
plot2_df = pd.concat([target_df, control_df, target_and_control_df], ignore_index=True)

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

# labels1 = [r'no $\mathcal{Q}$ / $f$ Ensemble', r'$\mathcal{Q}$ \ no $f$ Ensemble', r'$f$ Ensemble $+$ $\mathcal{Q}$']
subplot1_kwargs = {
    "hue" : 'label',
    "style" : 'label',
    # "dashes": [(4, 1), (2, 1), (1,1)]
    "palette": {labels1[0]: sns.color_palette()[0],
                labels1[1]: sns.color_palette()[1],
                labels1[2]: sns.color_palette()[4]},
    "dashes": {labels1[0]: (3,1),
               labels1[1]: (1,4.5),
               labels1[2]: (1,0),
               '': (0,0)},
}

# labels2 = ['Targets', 'Control', 'Targets $+$ Control']
subplot2_kwargs = {
    "hue" : 'label',
    "style" : 'label',
    # "dashes": [(4, 1), (2, 1), (1,1)]
    "palette": {labels2[0]: sns.color_palette()[2],
                labels2[1]: sns.color_palette()[3],
                labels2[2]: sns.color_palette()[4],
                '': None},
    "dashes": {labels2[0]: (3,1),
               labels2[1]: (1,4.5),
               labels2[2]: (1,0),
               '': (0,0)},
}

# fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(3.3,3.3), layout='constrained')
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(3.5,3.0))
plt.tight_layout()

# Plot on the first axis
axes[0] = sns.lineplot(plot1_df, ax=axes[0], x='global_step', y='median', **subplot1_kwargs)
for label in plot1_df['label'].unique():
    color = subplot1_kwargs["palette"][label]
    temp_df = plot1_df[plot1_df['label']==label]
    axes[0].fill_between(x=temp_df['global_step'], y1=temp_df['low'], y2=temp_df['high'], color=color, alpha=0.2)

axes[0].set_xlabel("")
axes[0].set_ylabel("")
axes[0].grid(True)
axes[0].legend()
sns.move_legend(axes[0], "lower center", title="MPPI Ingredients", bbox_to_anchor=(0.5, 1), frameon=False, ncol=1)

# Plot on the second axis
axes[1] = sns.lineplot(plot2_df, ax=axes[1], x='global_step', y='median', **subplot2_kwargs)
for label in plot2_df['label'].unique():
    color = subplot2_kwargs["palette"][label]
    temp_df = plot2_df[plot2_df['label']==label]
    axes[1].fill_between(x=temp_df['global_step'], y1=temp_df['low'], y2=temp_df['high'], color=color, alpha=0.2)

axes[1].set_xlabel("")
axes[1].grid(True)
axes[1].legend()
sns.move_legend(axes[1], "lower center", title="MPPI Usage", bbox_to_anchor=(0.5, 1), frameon=False, ncol=1)

# fine-tuning x-ticks
xticks = np.arange(0, 5*10e4+1, step=10e4)
yticks = np.arange(0, 3*10e2+1, step=5*10e1)
axes[0].set(xticks=xticks, yticks=yticks)
axes[1].set(xticks=xticks, yticks=yticks)
axes[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
axes[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
axes[0].xaxis.offsetText.set_visible(False)

fig.tight_layout(h_pad=0.5, w_pad=0.5) # need this to squeeze plots together
# add a big axis, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Time step")
plt.ylabel("Cumulative reward", labelpad=10)

plt.savefig('plotting/hop_eval.png', bbox_inches='tight')
plt.savefig('plotting/hop_eval.pdf', bbox_inches='tight')