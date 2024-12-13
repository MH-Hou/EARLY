import numpy as np
from time import sleep
import os


from scripts.joystick import Joystick


def main():
    task_name = 'nav_1'
    random_goal = False
    random_y = False
    max_trial_num = 1000
    success_num = 0
    success_thres = 5

    print("[Human Training Session]: Start training session ... ")

    env_info = {'random_goal': random_goal, 'random_y': random_y}
    joystick = Joystick(env_info=env_info)
    joystick.env.render_env.draw_text('Please wait for instruction ...')

    for demo_num in range(max_trial_num):
        joystick.env.render_env.draw_text('[Demo {}: Please provide a demo]'.format(demo_num + 1))

        init_x = np.random.uniform(0.1, 19.9)
        init_y = 4.0
        starting_pos = np.array([init_x, init_y])
        goal_pos = np.array([10.0, 16.0])

        oracle_states, oracle_actions = joystick.provide_joystick_demo(starting_pos=starting_pos, goal_pos=goal_pos)

        final_position = oracle_states[-1, 2:4]
        if np.linalg.norm(final_position - goal_pos) <= 1.0:
            print("Success!")
            success_num += 1

            if success_num >= success_thres:
                print("Great! You passed the training session!")
                break
        else:
            print("Failed!")
            success_num = 0

        joystick.env.render_env.clean()
        if demo_num < (max_trial_num - 1):
            joystick.env.render_env.draw_text('Please wait for instruction ...')
        else:
            joystick.env.render_env.draw_text('Great! All demos are provided!')

        sleep(2.0)

    joystick.stop()
    print('All finished')


if __name__ == '__main__':
    main()