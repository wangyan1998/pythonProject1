import math
import torch
import numpy
import csv
from scipy.optimize import fsolve

f = open('dataset/mydataset.csv', 'r')
reader = csv.reader(f)  # 创建一个与该文件相关的阅读器
result = list(reader)
inf = []
last = 0
attpos = [-2279827.6066, 5004703.5738, 3219775.4338]


# 从文件中获取数据
def getdata(i):
    global inf
    k = i
    j = i + 3
    while k < j:
        a = [float(result[k][2]), float(result[k][3]), float(result[k][4]), float(result[k][5]), float(result[k][6])]
        inf.append(a)
        k += 1
    print(inf)
    return inf


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


def print_revpos1():
    so = fsolve(get_position, [0, 0, 0])
    print("计算得到接收机位置", so)
    return so


def get_distance1(pos1):
    v1 = numpy.array([pos1[0], pos1[1], pos1[2]])
    v2 = numpy.array(attpos)
    distance = numpy.linalg.norm(v1 - v2)
    return distance


def getgrad(x, y, z):
    k = math.sqrt((x - attpos[0]) ** 2 + (y - attpos[1]) ** 2 + (z - attpos[2]) ** 2)
    n1 = x - attpos[0]
    n2 = y - attpos[1]
    n3 = z - attpos[2]
    m1 = x - inf[0][0]
    m2 = x - inf[1][0]
    m3 = x - inf[2][0]
    print("XYZ差距：", n1, n2, n3)
    res = []
    res.append(n1)
    res.append(n2)
    res.append(n3)
    # print(res)
    t = torch.tensor(res)
    sign = t.sign()
    return list(numpy.array(sign))


def inter():
    global inf
    global last
    last = 1000000
    dis = 1000000
    diff = 10000
    while diff > 0.001:
        r = print_revpos1()
        dis = get_distance1(r)
        print("距离差：", dis)
        if dis <= last:
            diff = abs(dis - last)
            last = dis
            sign = getgrad(r[0], r[1], r[2])
            # sign[0] = -sign[0]
            # sign[1] = -sign[1]
            # sign[2] = -sign[2]
            print(sign)
            i = 0
            while i < 3:
                inf[i][3] = inf[i][3] + 1 * sign[i]
                i = i + 1
        else:
            break


getdata(12)
inter()
