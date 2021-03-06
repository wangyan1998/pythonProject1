import math

from scipy.optimize import fsolve

inf = [[-368461.739, 26534822.568, -517664.322, 21966884.2427, -0.000104647296],
       [10002180.758, 12040222.131, 21796269.831, 23447022.1136, -0.000308443058],
       [-7036480.928, 22592611.906, 11809485.040, 20154521.4618, -0.000038172460],
       [8330122.383, 23062955.206, 10138101.681, 22129309.3677, -0.000239383603]]


def get_position(unsolved_value):
    x, y, z = unsolved_value[0], unsolved_value[1], unsolved_value[2]
    a = (x - inf[0][0]) ** 2 + (y - inf[0][1]) ** 2 + (z - inf[0][2]) ** 2
    b = (x - inf[1][0]) ** 2 + (y - inf[1][1]) ** 2 + (z - inf[1][2]) ** 2
    c = (x - inf[2][0]) ** 2 + (y - inf[2][1]) ** 2 + (z - inf[2][2]) ** 2
    return [
        math.sqrt(a) - inf[0][3] - (3 * 10 ** 8) * inf[0][4],
        math.sqrt(b) - inf[1][3] - (3 * 10 ** 8) * inf[1][4],
        math.sqrt(c) - inf[2][3] - (3 * 10 ** 8) * inf[2][4]
    ]


solve = fsolve(get_position, [0, 0, 0])
print(solve)
