import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

WINDOW = 500
TOTAL_LEN = 50_000
FINAL_LEN = 50_000

project = 'sac_baseline'
runs_df_sac = pd.read_pickle(f"data/{project}_all_data.pkl")
runs_df_sac = runs_df_sac[(runs_df_sac['env_id']=='InvertedDoublePendulum-v5')]
runs_df_sac = pd.concat([runs_df_sac], ignore_index=True)
runs_df_sac['label'] = "SAC"
runs_df_sac['label'] = runs_df_sac['label'].astype("category")

project = 'dual_mpcritic_ddpg_new_value_with_discount'

runs_df = pd.read_pickle(f"data/{project}_all_data.pkl")
print(runs_df['episodic_return'].iloc[0])

data_env = runs_df[(runs_df['env_id']=='InvertedDoublePendulum-v5')]
data_target_r20 = data_env[(data_env['num_target_rollouts']==20)]
data_target_h1 = data_target_r20[(data_target_r20['target_horizon']==1)]
data_target_h4 = data_target_r20[(data_target_r20['target_horizon']==4)]
data = pd.concat([data_target_h4], ignore_index=True)
# data = pd.concat([data_target_h4.sample(n=20)], ignore_index=True)

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

def process_data(data, y_column, label_columns, labels, total_len=500_000):
    """
    for label, col in zip(label, label_column):
        temp_data = data[data[label_column] == label]
    """
    for label, col in zip(labels, label_columns):
        data = data[data[col] == label]
    data = pad_arrays(data, y_column, total_len)
    med, low, high = median_low_high(data, y_column)
    med, low, high = map(moving_average, [med, low, high])
    med, low, high = map(trim, [med, low, high])
    return pd.DataFrame({
        'global_step' : ma_global_step(),
        'median': med,
        'low': low,
        'high': high,
        **{col: label for label, col in zip(labels, label_columns)}
    })

data.loc[data['mppi_target_warmstart'] == True, 'mppi_target_iterations'] = '1 (Warm)'
data.loc[data['mppi_target_warmstart'] == False, 'mppi_target_warmstart'] = 'Cold'
data.loc[data['mppi_target_warmstart'] == True, 'mppi_target_warmstart'] = 'Warm'
data.loc[~pd.isna(data['transition_ensemble_size']), 'transition_ensemble_size'] = 'Single'
data.loc[pd.isna(data['transition_ensemble_size']), 'transition_ensemble_size'] = 'Ensemble'

labels1 = ['1 (Warm)', 1, 5]
labels2 = ['Ensemble', 'Single']

dfs = {
    label1 : {
        label2 : None for label2 in labels2
    } for label1 in labels1
}

for label1 in labels1:
    for label2 in labels2:
        dfs[label1][label2] = process_data(
            data,
            'episodic_return',
            ['mppi_target_iterations', 'transition_ensemble_size'],
            [label1, label2]
        )
        dfs[label1][label2]['mppi_target_warmstart'] = 'Warm' if label1 == '1 (Warm)' else 'Cold'

plot_df = pd.DataFrame(columns = dfs[labels1[0]][labels2[0]].columns)
for label1 in labels1:
    for label2 in labels2:
        plot_df = pd.concat([plot_df, dfs[label1][label2]], ignore_index=True)

temp_df = process_data(runs_df_sac, 'episodic_return', ['label'], ["SAC"])
sac_median_final_value = temp_df['median'].iloc[-1]

sns.set(palette='husl', style='ticks')

SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    #  fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('xtick.major', size=5, width=1)    #  fontsize of the tick labels
plt.rc('ytick.major', size=5, width=1)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('legend', title_fontsize=SMALL_SIZE, fontsize=SMALL_SIZE)  # fontsize of the figure title
plt.rc('lines', linewidth=2.5)

params = {
        "text.usetex" : True,
        "font.family" : "serif",
        "font.serif" : ["Computer Modern Serif"]}
plt.rcParams.update(params)

plot_kwargs = {
    "x" : "global_step",
    "y" : "median",
    "hue" : 'mppi_target_iterations',
    "style" : 'mppi_target_iterations',
    "alpha" : 1.0
}

g = sns.FacetGrid(plot_df, col="mppi_target_warmstart", row="transition_ensemble_size", margin_titles=True, despine=False, legend_out=False, height=1.75, aspect=1)
for ax in g.axes.flatten():
    ax.axhline(y=sac_median_final_value, color="grey", dashes=(3,1), lw=1.5, label='')
g.map_dataframe(sns.lineplot, **plot_kwargs)
# g.set_axis_labels("Time step", "Reward")
for ax in g.axes.flat:
    ax.grid(True)
g.set_axis_labels("", "")
g.set_titles(col_template="{col_name}-start", row_template=r"{row_name} $f$", size=SMALL_SIZE)
g.tight_layout() # call before making legend outside
g.add_legend()
sns.move_legend(g, "lower center", title="MPPI target iterations", bbox_to_anchor=(0.5, 0.95), frameon=False, ncol=3)

for label1 in labels1:
    for label2 in labels2:
        temp_df = dfs[label1][label2]
        if label1 == '1 (Warm)':
            ax = g.axes[0][0] if label2 == 'Ensemble' else g.axes[1][0]
        else:
            ax = g.axes[0][1] if label2 == 'Ensemble' else g.axes[1][1]
        # for label in plot1_df['label'].unique():
            # color = subplot1_kwargs["palette"][label]
            # temp_df = plot1_df[plot1_df['label']==label]
            # axes[0].fill_between(x=temp_df['global_step'], y1=temp_df['low'], y2=temp_df['high'], color=color, alpha=0.2)
        ax.fill_between(x=temp_df['global_step'], y1=temp_df['low'], y2=temp_df['high'], alpha=0.2)

# fine-tuning x-ticks
xticks=np.arange(0, 5*10e3+1, step=10e3)
yticks=np.arange(0, 10e3, step=2*10e2)
g.set(xticks=xticks, yticks=yticks)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
bot_left_ax = list(g.axes.flat)[-2]
bot_left_ax.xaxis.offsetText.set_visible(False)

# need this to squeeze plots together
g.tight_layout(h_pad=0.5, w_pad=0.5)
# add a big axis, hide frame
# sns.set(style="ticks")
g.figure.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Time step")
plt.ylabel("Cumulative reward", labelpad=10)

# axes[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
plt.savefig('plotting/dip_eval.png', bbox_inches='tight')
plt.savefig('plotting/dip_eval.pdf', bbox_inches='tight')