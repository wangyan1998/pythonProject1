import numpy as np

inf = [[-368461.739, 26534822.568, -517664.322, 21966984.2427, -0.000104647296],
       [10002180.758, 12040222.131, 21796269.831, 23447022.1136, -0.000308443058],
       [-7036480.928, 22592611.906, 11809485.040, 20154521.4618, -0.000038172460],
       [8330122.41, 23062955.196, 10138101.718, 22129309.3677, -0.000239356]]

inf1 = [[-368461.739, 26534822.568, -517664.322],
        [10002180.758, 12040222.131, 21796269.831],
        [-7036480.928, 22592611.906, 11809485.040],
        [8330122.41, 23062955.196, 10138101.718]]

p = [[21966984.2427], [23447022.1136], [20154521.4618], [22129309.3677]]

t = [[-0.000104647296], [-0.000308443058], [-0.000038172460], [-0.000239356]]

pos0 = [0, 0, 0]

pos1 = [-2279829.1069, 5004709.2387, 3219779.0559]


# 获取detp，也就是以第一个伪距为基准的伪距差
def getdetp():
    res = []
    k = len(p)
    i = 1
    while i < k:
        res.append(p[i][0] - p[0][0])
        i = i + 1
    # print(res)
    return res


# 求两点之间的距离
def get_distance1(pos1, pos2):
    v1 = np.array([pos1[0], pos1[1], pos1[2]])
    v2 = np.array([pos2[0], pos2[1], pos2[2]])
    distance = np.linalg.norm(v1 - v2)
    return distance


# 获得每一个r
def getdis():
    res = []
    k = len(p)
    i = 0
    while i < k:
        dis = get_distance1(inf[i], pos0)
        res.append(dis)
        i = i + 1
    # print(res)
    return res


# 获得r的差
def getdetdis(dis):
    res = []
    k = len(dis)
    i = 1
    while i < k:
        res.append(dis[i] - dis[0])
        i = i + 1
    # print(res)
    return res


def getmatH(info, pos, r):
    res = []
    k = len(info)
    i = 1
    c = 3 * 10 ** 8
    while i < k:
        l = []
        l.append(((pos[0] - info[i][0]) / r[i]) - ((pos[0] - info[0][0]) / r[0]))
        l.append(((pos[1] - info[i][1]) / r[i]) - ((pos[1] - info[0][1]) / r[0]))
        l.append(((pos[2] - info[i][2]) / r[i]) - ((pos[2] - info[0][2]) / r[0]))
        i = i + 1
        res.append(l)
    print(res)
    return res


detp = getdetp()
dis = getdis()
print("伪距：\n", dis)
detdis = getdetdis(dis)
print("伪距差：\n", detdis)
H = getmatH(inf1, pos0, dis)
print("观测矩阵为H：\n", np.array(H))
H1 = np.linalg.pinv(H)  # 矩阵求逆
print("观测矩阵求逆H1：\n", H1)
H2 = np.transpose(H)  # 矩阵转置
print("观测矩阵转置H2：\n", H2)
H3 = np.dot(H2, H)
print("中间结果,H3：\n", H3)
H4 = np.linalg.pinv(H3)
print("中间结果,H4：\n", H4)
# H5 = np.dot(H4, H2)
# print("中间结果,H5：\n", np.array(H5))
Ho = np.dot(H, H4)
H6 = np.dot(Ho, H2)
print("中间结果,H6：\n", H6)
I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
S = I - H6
print("中间结果,S:\n", S)
R = np.dot(S, detp)
print("中间结果,R：\n", R)
RT = np.transpose(R)
Ts = np.dot(RT, R)
print("Ts为：\n", Ts)
