import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



def plot_learning_curve(df, type, hue, name):
    sns.set_style('whitegrid')
    colors = ["#c44e53", "#4c72b0"]
    # colors = ["#ff1400", "#06ffe2", "#0000ff", "#850085", "#008100"]
    # sns.set_palette(sns.color_palette(colors))
    if type == 'per_step':
        # f = sns.lineplot(data=df, x="step", y="reward", hue=hue, errorbar=('ci', 95), palette=sns.color_palette("rocket"))
        f = sns.lineplot(data=df, x="step", y="reward", hue=hue, ci='sd', palette=sns.color_palette(colors))
    else:
        # f = sns.lineplot(data=df, x="demo", y="reward", hue=hue, errorbar=('ci', 95), palette=sns.color_palette("rocket"))
        f = sns.lineplot(data=df, x="demo", y="reward", hue=hue, ci='sd',
                         palette=sns.color_palette("rocket"))

    # f.axes.axhline(-200, c='grey', ls='--', label="gail")
    # f.axes.axhline(-675.97, c='red', ls='--', label="bc")

    # labels = ["active_sac", "active_sac_random"]
    # handles, _ = f.get_legend_handles_labels()
    #
    # # Slice list to remove first handle
    # plt.legend(handles=handles[:], labels=labels)

    # plt.savefig(str(name) + '.png', dpi=300)
    plt.show()
    # plt.savefig('1.png', dpi=300)


def plot_reward_and_success_rate(reward_df, success_df, hue, plot_name, expert_average_reward, expert_success_rate):
    sns.set_style('darkgrid')
    # colors = ["#4c72b0", "#06ffe2", '#ff8800', '#008100', '#850085', "#ff1400"]
    colors = ["#4c72b0", "#06ffe2", "#ff1400", '#ff8800']
    # labels = ['B-ARLD', 'R-ARLD', 'P-RLD', 'SAC', 'PH-RLD', 'E-ARLD', 'Expert']
    labels = ['DDPG-LfD', 'I-ARLD' ,'EARLY', 'GAIL', 'Expert']

    # plot average episode reward
    f1 = sns.lineplot(data=reward_df, x="step", y="reward", hue=hue, errorbar=('sd'), palette=sns.color_palette(colors))
    f1.axes.axhline(expert_average_reward, c='blue', ls='--', label="expert")
    f1.set(xlabel='environment steps', ylabel='average episode rewards')

    handles, _ = f1.get_legend_handles_labels()
    # Slice list to remove first handle
    plt.legend(handles=handles[:], labels=labels)
    plt.xlim(0, int(100 * 1e3))

    plt.savefig('plots/' + 'reward_' + plot_name + '.png', dpi=300)
    plt.show()

    # plot average episode success rate
    f2 = sns.lineplot(data=success_df, x="step", y="success_rate", hue=hue, errorbar=('sd'), palette=sns.color_palette(colors))
    f2.axes.axhline(expert_success_rate, c='blue', ls='--', label="expert")
    f2.set(xlabel='environment steps', ylabel='average success rate')

    handles, _ = f1.get_legend_handles_labels()
    # Slice list to remove first handle
    plt.legend(handles=handles[:], labels=labels)
    plt.xlim(0, int(100 * 1e3))

    plt.savefig('plots/' + 'success_rate_' + plot_name + '.png', dpi=300)
    plt.show()


def smooth(data, sm=1):
    smooth_data = []
    if sm > 1:
        for d in data:
            z = np.ones(len(d))
            y = np.ones(sm)*1.0
            d = np.convolve(y, d, "same")/np.convolve(y, z, "same")
            smooth_data.append(d)
    else:
        smooth_data = data
    return smooth_data


def smooth_new(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed

def main_new():
    methods = ['DDPGfD', 'I-ARLD' ,'E-ARLD', 'GAIL']
    original_method_names = ['ddpg_lfd', 'isolated_active', 'active_sac', 'gail']
    task_names = ['nav_1', 'nav_2', 'nav_3']
    max_demo_num = 60
    uncertainty_method = 'td_error'
    smooth_fac = 10

    expert_rewards_list = [983.12, 986.13, 986.97]
    expert_success_list = [1.0, 1.0, 1.0]

    for task_id in range(len(task_names)):
        task_name = task_names[task_id]
        step_reward_data_dict = {'step': [], 'reward': [], 'method': []}
        step_success_data_dict = {'step': [], 'success_rate': [], 'method': []}

        expert_average_rewards = expert_rewards_list[task_id]
        expert_success_rate = expert_success_list[task_id]

        if task_name == 'nav_1':
            ratio = 0.35
        elif task_name == 'nav_2':
            ratio = 0.3
        else:
            ratio = 0.5

        for method_id in range(len(methods)):
            method = methods[method_id]
            original_saving_name = original_method_names[method_id]

            res_path_list = []
            if original_saving_name == 'sac_lfd_human':
                total_subject_num = 5
                for i in range(total_subject_num):
                    subject_id = i + 1
                    res_path = 'evaluation_res/' + task_name + '/' + original_saving_name + '/' + 'sub_' + str(subject_id) + '/max_demo_' + str(
                        max_demo_num) + '/ratio_' + str(
                        ratio) + '/' + uncertainty_method + '/'
                    res_path_list.append(res_path)
            else:
                if original_saving_name == 'active_sac':
                    res_path = 'evaluation_res/new/' + task_name + '/' + original_saving_name + '/max_demo_' + str(
                        max_demo_num) + '/ratio_' + str(
                        ratio) + '/' + uncertainty_method + '/'
                    res_path_list.append(res_path)
                elif original_saving_name == 'ddpg_lfd':
                    res_path = 'evaluation_res/new/' + task_name + '/' + original_saving_name + '/max_demo_' + str(
                        max_demo_num) + '/'
                    res_path_list.append(res_path)
                elif original_saving_name == 'isolated_active':
                    res_path = 'evaluation_res/new/' + task_name + '/' + original_saving_name + '/max_demo_' + str(
                        max_demo_num) + '/ratio_' + str(0.1) + '/'
                    res_path_list.append(res_path)
                else:
                    res_path = 'evaluation_res/new/' + task_name + '/' + original_saving_name + '/max_demo_' + str(
                        max_demo_num) + '/'
                    res_path_list.append(res_path)

            for res_path in res_path_list:
                res_per_step = np.genfromtxt(res_path + 'res_per_step_new.csv', delimiter=' ')
                for seed_id in range(1, res_per_step.shape[1]):
                    all_steps_data = res_per_step[:, seed_id]
                    # smoothed_data = smooth(data=[all_steps_data], sm=smooth_fac)[0] # 1d np array
                    smoothed_data = np.array(smooth_new(all_steps_data, 0.6))
                    for j in range(smoothed_data.shape[0]):
                        step = res_per_step[j][0]
                        smoothed_reward = smoothed_data[j]
                        step_reward_data_dict['step'].append(step)
                        step_reward_data_dict['reward'].append(smoothed_reward)
                        step_reward_data_dict['method'].append(method)

                success_res_per_step = np.genfromtxt(res_path + 'success_res_per_step_new.csv', delimiter=' ')
                for seed_id in range(1, success_res_per_step.shape[1]):
                    all_steps_data = success_res_per_step[:, seed_id]
                    # smoothed_data = smooth(data=[all_steps_data], sm=smooth_fac)[0]  # 1d np array
                    smoothed_data = np.array(smooth_new(all_steps_data, 0.6))
                    for j in range(smoothed_data.shape[0]):
                        step = res_per_step[j][0]
                        smoothed_success = smoothed_data[j]
                        step_success_data_dict['step'].append(step)
                        step_success_data_dict['success_rate'].append(smoothed_success)
                        step_success_data_dict['method'].append(method)

        step_reward_df = pd.DataFrame(data=step_reward_data_dict)
        step_success_df = pd.DataFrame(data=step_success_data_dict)
        plot_reward_and_success_rate(reward_df=step_reward_df, success_df=step_success_df, hue='method', plot_name=task_name, expert_average_reward=expert_average_rewards, expert_success_rate=expert_success_rate)


def main():
    methods = ['B-ARLD', 'R-ARLD', 'P-RLD', 'SAC', 'PH-RLD', 'E-ARLD']
    original_method_names = ['active_sac_bern_timing', 'active_sac_stream_query', 'sac_lfd', 'sac', 'sac_lfd_human', 'active_sac']
    task_names = ['nav_1', 'nav_2', 'nav_3']
    max_demo_num = 60
    uncertainty_method = 'td_error'
    expert_rewards_list = [983.12, 986.13, 986.97]
    expert_success_list = [1.0, 1.0, 1.0]

    for i in range(len(task_names)):
        task_name = task_names[i]
        step_reward_data_dict = {'step': [], 'reward': [], 'method': [], 'ratio': []}
        step_success_data_dict = {'step': [], 'success_rate': [], 'method': [], 'ratio': []}

        expert_average_rewards = expert_rewards_list[i]
        expert_success_rate = expert_success_list[i]

        if task_name == 'nav_3':
            ratio = 0.4
        else:
            ratio = 0.3

        for i in range(len(methods)):
            method = methods[i]
            original_saving_name = original_method_names[i]

            res_path_list = []
            if original_saving_name == 'sac_lfd_human':
                total_subject_num = 5
                for i in range(total_subject_num):
                    subject_id = i + 1
                    res_path = 'evaluation_res/' + task_name + '/' + original_saving_name + '/' + 'sub_' + str(subject_id) + '/max_demo_' + str(
                        max_demo_num) + '/ratio_' + str(
                        ratio) + '/' + uncertainty_method + '/'
                    res_path_list.append(res_path)
            else:
                res_path = 'evaluation_res/' + task_name + '/' + original_saving_name + '/max_demo_' + str(
                    max_demo_num) + '/ratio_' + str(
                    ratio) + '/' + uncertainty_method + '/'
                res_path_list.append(res_path)

            for res_path in res_path_list:
                res_per_step = np.genfromtxt(res_path + 'res_per_step_new.csv', delimiter=' ')
                for i in range(np.shape(res_per_step)[0]):
                    step = res_per_step[i][0]
                    for j in range(1, np.shape(res_per_step)[1]):
                        reward = res_per_step[i][j]
                        step_reward_data_dict['step'].append(step)
                        step_reward_data_dict['reward'].append(reward)
                        step_reward_data_dict['method'].append(method)
                        step_reward_data_dict['ratio'].append(ratio)

                success_res_per_step = np.genfromtxt(res_path + 'success_res_per_step_new.csv', delimiter=' ')
                for i in range(np.shape(success_res_per_step)[0]):
                    step = success_res_per_step[i][0]
                    for j in range(1, np.shape(success_res_per_step)[1]):
                        success = success_res_per_step[i][j]
                        step_success_data_dict['step'].append(step)
                        step_success_data_dict['success_rate'].append(success)
                        step_success_data_dict['method'].append(method)
                        step_success_data_dict['ratio'].append(ratio)

        step_reward_df = pd.DataFrame(data=step_reward_data_dict)
        step_success_df = pd.DataFrame(data=step_success_data_dict)

        plot_reward_and_success_rate(reward_df=step_reward_df, success_df=step_success_df, hue='method', plot_name=task_name, expert_average_reward=expert_average_rewards, expert_success_rate=expert_success_rate)



def main_old():
    methods = ['active_sac_20','active_sac_random_20', 'active_sac_200', 'active_sac_random_200']
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    step_data_dict = {'step':[], 'reward':[], 'method':[], 'ratio':[]}
    demo_data_dict = {'demo':[], 'reward':[], 'method':[], 'ratio':[]}

    method = 'active_sac'
    max_demo_num = 60
    # ratio = 0.2
    uncertainty_method = 'td_error'

    # for method in methods:
    for ratio in ratios:
        res_path = 'evaluation_res/' + method + '/max_demo_' + str(max_demo_num) + '/ratio_' + str(
        ratio) + '/' + uncertainty_method + '/'

        res_per_step = np.genfromtxt(res_path + 'res_per_step.csv',
                                     delimiter=' ')
        res_per_demo = np.genfromtxt(res_path + 'res_per_demo.csv',
                                     delimiter=' ')

        for i in range(np.shape(res_per_step)[0]):
            step = res_per_step[i][0]
            for j in range(1, np.shape(res_per_step)[1]):
                reward = res_per_step[i][j]
                step_data_dict['step'].append(step)
                step_data_dict['reward'].append(reward)
                step_data_dict['method'].append(method)
                step_data_dict['ratio'].append(ratio)

        for i in range(np.shape(res_per_demo)[0]):
            demo_id = res_per_demo[i][0]
            for j in range(1, np.shape(res_per_demo)[1]):
                reward = res_per_demo[i][j]
                demo_data_dict['demo'].append(demo_id)
                demo_data_dict['reward'].append(reward)
                demo_data_dict['method'].append(method)
                demo_data_dict['ratio'].append(ratio)

    step_df = pd.DataFrame(data=step_data_dict)
    demo_df = pd.DataFrame(data=demo_data_dict)

    plot_learning_curve(step_df, type='per_step', hue='ratio', name=4)
    plot_learning_curve(demo_df, type='per_demo', hue='ratio', name=5)


if __name__=='__main__':
    main_new()

