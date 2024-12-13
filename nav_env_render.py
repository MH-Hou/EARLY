import numpy as np
import PyQt5
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets
from time import sleep


class dummy_nav_render():
    def __init__(self, goal=(10,16), with_grid=True, task_name='nav_1'):
        self.app = QtWidgets.QApplication([])
        self.window = gl.GLViewWidget()

        camera_params = {'rotation': PyQt5.QtGui.QQuaternion(1.0, 0.0, 0.0, 0.0),
                         'distance': 35.0,
                         'fov': 50}
        self.window.setCameraParams(**camera_params)
        self.window.pan(dx=10, dy=10, dz=0)

        # self.window.opts['distance'] = 200
        # self.window.opts['fov'] = 1

        self.window.show()  # show the window

        self.goal = np.array([goal[0], goal[1]])
        goal_points_drawing_var = gl.GLScatterPlotItem(pos=self.goal, color=(1.0, 1.0, 1.0, 1.0), size=10.0)
        self.window.addItem(goal_points_drawing_var)

        axis = gl.GLAxisItem()
        self.window.addItem(axis)
        x_label = gl.GLTextItem(pos=[1, 0, 0], text='X')
        self.window.addItem(x_label)
        y_label = gl.GLTextItem(pos=[0, 1, 0], text='Y')
        self.window.addItem(y_label)
        z_label = gl.GLTextItem(pos=[0, 0, 1], text='Z')
        # self.window.addItem(z_label)

        # add grid to the visualization
        if with_grid:
            grid = gl.GLGridItem()
            grid.translate(dx=10, dy=10, dz=0)
            # grid.setSize(x=40,y=40,z=0)
            self.window.addItem(grid)

        if task_name == 'nav_1':
            starting_line = np.array([[0, 4], [20, 4]])
            drawing_var = gl.GLLinePlotItem(pos=starting_line, width=0.5, antialias=False, color='green')
            # self.window.addItem(drawing_var)
        elif task_name == 'nav_2':
            starting_line = np.array([[0, 4], [20, 4]])
            drawing_var = gl.GLLinePlotItem(pos=starting_line, width=0.5, antialias=False, color='green')
            self.window.addItem(drawing_var)

            goal_line = np.array([[0, 16], [20, 16]])
            drawing_var = gl.GLLinePlotItem(pos=goal_line, width=0.5, antialias=False, color='white')
            self.window.addItem(drawing_var)
        else:
            starting_line_1 = np.array([[0, 1], [20, 1]])
            drawing_var = gl.GLLinePlotItem(pos=starting_line_1, width=0.5, antialias=False, color='green')
            self.window.addItem(drawing_var)

            starting_line_2 = np.array([[0, 8], [20, 8]])
            drawing_var = gl.GLLinePlotItem(pos=starting_line_2, width=0.5, antialias=False, color='green')
            self.window.addItem(drawing_var)

            goal_line_1 = np.array([[0, 12], [20, 12]])
            drawing_var = gl.GLLinePlotItem(pos=goal_line_1, width=0.5, antialias=False, color='white')
            self.window.addItem(drawing_var)

            goal_line_2 = np.array([[0, 19], [20, 19]])
            drawing_var = gl.GLLinePlotItem(pos=goal_line_2, width=0.5, antialias=False, color='white')
            self.window.addItem(drawing_var)

        self.draw_boundaries()
        self.draw_obstacles()

        self.app.processEvents()
        sleep(0.1)

        self.last_drawing_vars = []
        self.last_drawing_vars_goal = [goal_points_drawing_var]
        # self.last_drawing_vars_goal =[]
        self.last_drawing_vars_demo_num = []
        self.last_drawing_vars_text = []

        # simple hack to avoid the initial black screen
        self.draw_current_position(pos=np.array([0, 0, 0]), alpha=0.0)

    def draw_boundaries(self):
        vertex_1 = [0, 0]
        vertex_2 = [0, 20]
        vertex_3 = [20, 20]
        vertex_4 = [20, 0]

        line_1_pts = np.array([vertex_1, vertex_2])
        line_2_pts = np.array([vertex_2, vertex_3])
        line_3_pts = np.array([vertex_3, vertex_4])
        line_4_pts = np.array([vertex_4, vertex_1])

        drawing_var = gl.GLLinePlotItem(pos=line_1_pts, width=5, antialias=False, color='blue')
        self.window.addItem(drawing_var)

        drawing_var = gl.GLLinePlotItem(pos=line_2_pts, width=5, antialias=False, color='blue')
        self.window.addItem(drawing_var)

        drawing_var = gl.GLLinePlotItem(pos=line_3_pts, width=5, antialias=False, color='blue')
        self.window.addItem(drawing_var)

        drawing_var = gl.GLLinePlotItem(pos=line_4_pts, width=5, antialias=False, color='blue')
        self.window.addItem(drawing_var)

    def draw_obstacles(self):
        obstacles_pos_list = [[[5, 10], [13, 10]],
                              [[15, 10], [17, 10]]] # in the form of (num_of_obstacles, num_of_points, pos_dimensions)

        for obstacle in obstacles_pos_list:
            left_side_pos = obstacle[0]
            left_side_pos = (left_side_pos[0], left_side_pos[1], 0) # 3d  array of (x, y, z)
            right_side_pos = obstacle[1]
            right_side_pos = (right_side_pos[0], right_side_pos[1], 0)
            points = np.array([left_side_pos, right_side_pos])

            drawing_var = gl.GLLinePlotItem(pos=points, width=5, antialias=False, color='red')
            self.window.addItem(drawing_var)

    def draw_current_position(self, pos, alpha=1.0):
        # clean the last plotting lines
        for last_drawing_var in self.last_drawing_vars:
            self.window.removeItem(last_drawing_var)
        self.last_drawing_vars = []

        points_drawing_var = gl.GLScatterPlotItem(pos=pos, color=(0.0, 1.0, 0.0, alpha), size=10.0)
        self.window.addItem(points_drawing_var)
        self.last_drawing_vars.append(points_drawing_var)

        self.app.processEvents()

    def draw_current_goal_position(self, pos):
        # clean the last plotting lines
        for last_drawing_var in self.last_drawing_vars_goal:
            self.window.removeItem(last_drawing_var)
        self.last_drawing_vars_goal = []

        points_drawing_var = gl.GLScatterPlotItem(pos=pos, color=(1.0, 1.0, 1.0, 0.6), size=25.0)
        self.window.addItem(points_drawing_var)
        self.last_drawing_vars_goal.append(points_drawing_var)

        self.app.processEvents()

    def draw_current_demo_num(self, demo_num):
        # clean the last plotting lines
        for last_drawing_var in self.last_drawing_vars_demo_num:
            self.window.removeItem(last_drawing_var)
        self.last_drawing_vars_demo_num = []

        demo_text = gl.GLTextItem(pos=[-5, 20, 0], text='Demo ' + str(demo_num))
        self.window.addItem(demo_text)
        self.last_drawing_vars_demo_num.append(demo_text)

        self.app.processEvents()

    def draw_text(self, text):
        # clean last text drawing
        for last_drawing_var in self.last_drawing_vars_text:
            self.window.removeItem(last_drawing_var)
        self.last_drawing_vars_text = []

        text_var = gl.GLTextItem(pos=[-5, 20, 0], text=text)
        self.window.addItem(text_var)
        self.last_drawing_vars_text.append(text_var)

        self.app.processEvents()

    def clean(self):
        # clean last position drawing
        for last_drawing_var in self.last_drawing_vars:
            self.window.removeItem(last_drawing_var)
        self.last_drawing_vars = []

        # clean last goal drawing
        for last_drawing_var in self.last_drawing_vars_goal:
            self.window.removeItem(last_drawing_var)
        self.last_drawing_vars_goal = []

    def stop_render(self):
        self.window.close()



if __name__ == '__main__':
    nav_render_env = dummy_nav_render()
    # sleep(0.01)
    # nav_render_env.draw_current_position(pos=np.array([0,0,0]))
    sleep(5.0)

    starting_point = np.array([0,0,0])
    end_point = np.array([10,10,0])

    for i in range(200):
        current_pos = starting_point + i/10.0
        current_pos[2] = 0.0
        nav_render_env.draw_current_position(pos=current_pos)
        print("pos [{}]".format(i + 1))
        sleep(0.05)

    # nav_render_env.stop_render()
    print("Test finished")