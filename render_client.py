import numpy as np
import redis
from time import sleep
import argparse


from nav_env_render import dummy_nav_render


class RenderClient:
    def __init__(self, task_name='nav_3'):
        self.r = redis.Redis(host='localhost', port=6379, db=0)
        self.starting_pos_subscriber = self.r.pubsub()
        self.starting_pos_subscriber.subscribe(**{'init_pos':self.init_pos_message_handler})
        self.goal_pos_subscriber = self.r.pubsub()
        self.goal_pos_subscriber.subscribe(**{'goal_pos': self.goal_pos_message_handler})
        self.stop_signal_subscriber = self.r.pubsub()
        self.stop_signal_subscriber.subscribe(**{'stop': self.stop_message_handler})
        self.demo_num_subscriber = self.r.pubsub()
        self.demo_num_subscriber.subscribe(**{'demo_num': self.demo_num_message_handler})

        self.whether_stop = False

        self.demo_num = 0

        self.default_starting_position = np.array([10.0, 4.0])
        self.default_goal_position = np.array([10.0, 16.0])
        self.starting_pos = self.default_starting_position.copy()
        self.goal_pos = self.default_goal_position.copy()

        self.env_render = dummy_nav_render(with_grid=False, task_name=task_name)
        # self.env_render.draw_current_position(pos=self.default_starting_position)

        self.starting_pos_thread = self.starting_pos_subscriber.run_in_thread(sleep_time=0.01)
        self.goal_pos_thread = self.goal_pos_subscriber.run_in_thread(sleep_time=0.01)
        self.stop_signal_thread = self.stop_signal_subscriber.run_in_thread(sleep_time=0.01)
        self.demo_num_thread = self.demo_num_subscriber.run_in_thread(sleep_time=0.01)

    def init_pos_message_handler(self, message):
        print("[Render Client]: Receive agent starting pos data: {}".format(message['data']))

        starting_pos = [float(i) for i in message['data'].split()]
        print(starting_pos)
        self.starting_pos = np.array(starting_pos)

    def goal_pos_message_handler(self, message):
        print("[Render Client]: Receive agent goal pos data: {}".format(message['data']))

        goal_pos = [float(i) for i in message['data'].split()]
        print(goal_pos)
        self.goal_pos = np.array(goal_pos)

    def stop_message_handler(self, message):
        print("[Render Client]: Receive stop signal: {}".format(message['data']))
        self.whether_stop = True

    def demo_num_message_handler(self, message):
        print("[Render Client]: Receive demo num: {}".format(message['data']))
        self.demo_num = int(message['data'])

    def render(self):
        self.env_render.draw_current_position(pos=self.starting_pos)
        self.env_render.draw_current_goal_position(pos=self.goal_pos)
        self.env_render.draw_current_demo_num(demo_num=self.demo_num)

    def stop(self):
        self.env_render.stop_render()
        print("[Render Client]: Stop env render window")

        self.starting_pos_thread.stop()
        self.goal_pos_thread.stop()
        self.stop_signal_thread.stop()
        self.demo_num_thread.stop()
        print("[Render Client]: Stop all subscriber threads")


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', help='id of nav task', default=1, type=int)

    return parser.parse_args()



def main():
    args = argparser()
    task_name = 'nav_' + str(args.task_id)

    render_client = RenderClient(task_name=task_name)

    while not render_client.whether_stop:
        render_client.render()

    render_client.stop()
    print("Main thread finished")


if __name__ == '__main__':
    main()

