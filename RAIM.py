import math

import numpy
from scipy.optimize import fsolve

inf = [[-368461.739, 26534822.568, -517664.322, 21966884.2427, -0.000104647296],
       [10002180.758, 12040222.131, 21796269.831, 23447022.1136, -0.000308443058],
       [-7036480.928, 22592611.906, 11809485.040, 20154521.4618, -0.000038172460]]

inf1 = [[-368461.739, 26534822.568, -517664.322, 21966884.2427, -0.000104647296],
        [10002180.758, 12040222.131, 21796269.831, 23447022.1136, -0.000308443058],
        [-7036480.928, 22592611.906, 11809485.040, 20154521.4618, -0.000038172460]]


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


def get_position1(unsolved_value):
    x, y, z = unsolved_value[0], unsolved_value[1], unsolved_value[2]
    a = (x - inf1[0][0]) ** 2 + (y - inf1[0][1]) ** 2 + (z - inf1[0][2]) ** 2
    b = (x - inf1[1][0]) ** 2 + (y - inf1[1][1]) ** 2 + (z - inf1[1][2]) ** 2
    c = (x - inf1[2][0]) ** 2 + (y - inf1[2][1]) ** 2 + (z - inf1[2][2]) ** 2
    return [
        math.sqrt(a) - inf1[0][3] - (3 * 10 ** 8) * inf1[0][4],
        math.sqrt(b) - inf1[1][3] - (3 * 10 ** 8) * inf1[1][4],
        math.sqrt(c) - inf1[2][3] - (3 * 10 ** 8) * inf1[2][4]
    ]


def get_distance1(pos1, pos2):
    v1 = numpy.array([pos1[0], pos1[1], pos1[2]])
    v2 = numpy.array([pos2[0], pos2[1], pos2[2]])
    distance = numpy.linalg.norm(v1 - v2)

    return distance


def get_distance2(pos1, pos2, clk):
    v1 = numpy.array([pos1[0], pos1[1], pos1[2]])
    v2 = numpy.array([pos2[0], pos2[1], pos2[2]])
    distance = numpy.linalg.norm(v1 - v2)
    res = distance - (3 * 10 ** 8) * clk[0]

    return res


solve = fsolve(get_position, [0, 0, 0])
print("接收机位置为：")
print(solve)
nextpos = fsolve(get_position1, [0, 0, 0])
print(nextpos)
# 冗余数据，pos是卫星的位置，clk是钟差，pseud是伪距，这些都可以直接从文件中获取
pos = [
    [8330122.383, 23062955.206, 10138101.681],
    [-11000923.071, 24101993.965, -4054413.007],
    [-16563409.228, 5239172.696, 20004668.669],
    [-18300882.387, -3909559.927, 19515733.472],
    [-23946891.397, -2616591.370, 11396400.893],
    [-12378889.546, 13423361.822, 19017004.492]
]
clk = [[-0.000239383603], [-0.000243993114], [-0.000172589899], [0.000013685894], [0.000228495598], [-0.000324999128]]
pseud = [[22129309.3677], [22292035.2737], [22092757.5787], [24525360.0865], [24311823.2505], [20650197.1218]]

dis = get_distance1(solve, nextpos)
print("攻击前后接收机位置差：")
print(dis)
i = 0
while i < 6:
    solve1 = get_distance2(solve, pos[i], clk[i])
    solve2 = get_distance2(nextpos, pos[i], clk[i])
    print("计算出的伪距为：")
    print(solve1)
    print(solve2)
    err = pseud[i] - solve1
    err1 = pseud[i] - solve2
    print("伪距的误差为（单位：m）:")
    print(err)
    print(err1)
    i = i + 1
