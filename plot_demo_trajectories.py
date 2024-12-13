import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors




def main():
    """ Load demo data """
    task_name = 'nav_3'
    method = 'active_sac'
    # method = 'isolated_active'
    max_demo_num = 60
    uncertainty_method = 'td_error'

    if method == 'active_sac':
        if task_name == 'nav_1':
            ratio = 0.35
        elif task_name == 'nav_2':
            ratio = 0.3
        else:
            ratio = 0.4
        demo_traj_data_path = 'demo_trajectory_data/' + task_name + '/' + method + '/max_demo_' + str(
            max_demo_num) + '/ratio_' + str(ratio) + '/' + uncertainty_method + '/'
    else:
        ratio = 0.1
        demo_traj_data_path = 'demo_trajectory_data/' + task_name + '/' + method + '/max_demo_' + str(
            max_demo_num) + '/ratio_' + str(ratio) + '/'

    # a list of 2d trajectories, each trajectory include a bunch of (x,y) points
    all_trajs = np.genfromtxt(demo_traj_data_path + 'demo_trajectory_data_new.csv', delimiter=' ')
    steps = all_trajs.shape[0]
    traj_id = 0
    all_trajs_cleaned = []
    traj = [] # a list of (x, y) points
    for step in range(steps):
        if all_trajs[step][0] == np.inf:
            # not to save when the traj is empty
            if traj_id > 0:
                traj = np.array(traj) # 2d np array
                all_trajs_cleaned.append(traj)
                # print("Finished loading trajectory {}".format(traj_id))

            traj = []
            # print("Start to load trajectory {}".format(traj_id + 1))
            traj_id += 1
            continue

        new_point = [all_trajs[step][0], all_trajs[step][1]]
        traj.append(new_point)

        # save the last traj
        if step == (steps - 1):
            traj = np.array(traj)  # 2d np array
            all_trajs_cleaned.append(traj)
            # print("Finished loading trajectory {}".format(traj_id))

    print("Finished loading all trajectories")
    # print(len(all_trajs_cleaned))

    fig, ax = plt.subplots(figsize=(6, 6))

    # plot the scene
    ax.plot([5.2, 12.8], [10, 10], c='black', linewidth=2)
    ax.plot([15.2, 16.8], [10, 10], c='black', linewidth=2)
    # ax.plot([0, 20], [0, 0], c='black')
    # ax.plot([0, 20], [20, 20], c='black')
    # ax.plot([0, 0], [0, 20], c='black')
    # ax.plot([20, 20], [0, 20], c='black')

    COLOR = (0, 0, 0)

    def color_conv(color_range):
        return (COLOR[0] + color_range, COLOR[1], COLOR[2])

    plt.xlim(0, 20)
    plt.ylim(0, 20)
    ax.set_aspect('equal', adjustable='box')
    for i in range(len(all_trajs_cleaned)):
        traj = all_trajs_cleaned[i]

        ax.plot(traj[:, 0], traj[:, 1], c='r', alpha=0.3 + i/(len(all_trajs_cleaned) * 10))

    plt.savefig('demo_' + task_name + '_' + method)
    plt.show()

if __name__ == '__main__':
    main()
