import math
import csv
import numpy
import xlwt
from scipy.optimize import fsolve
import readbit

inf = [[-368461.739, 26534822.568, -517664.322, 21966984.2427, -0.000104647296],
       [10002180.758, 12040222.131, 21796269.831, 23447022.1136, -0.000308443058],
       [-7036480.928, 22592611.906, 11809485.040, 20154521.4618, -0.000038172460]]

inf1 = [[-368461.739, 26534822.568, -517664.322, 21966974.2427, -0.000104647296],
        [10002180.758, 12040222.131, 21796269.831, 23447022.1136, -0.000308443058],
        [-7036480.928, 22592611.906, 11809485.040, 20154521.4618, -0.000038172460]]

info = [[-368461.739, 26534822.568, -517664.322, 21966984.2427, -0.000104647296],
        [10002180.758, 12040222.131, 21796269.831, 23447022.1136, -0.000308443058],
        [-7036480.928, 22592611.906, 11809485.040, 20154521.4618, -0.000038172460]]
inf4 = []
f = open('dataset/mydataset.csv', 'r')
reader = csv.reader(f)  # 创建一个与该文件相关的阅读器
result = list(reader)
num = 5


# 从文件中获取数据
def getdata(i):
    k = i
    j = i + 3
    res = []
    while k < j:
        a = [float(result[k][2]), float(result[k][3]), float(result[k][4]), float(result[k][5]), float(result[k][6])]
        res.append(a)
        k += 1
    return res


##############################################################################################
def get_putongdata(i):
    global inf
    inf = getdata(i)
    print("攻击前数据矩阵：", inf)


# 普通定位源码
def get_position(unsolved_value):
    x, y, z = unsolved_value[0], unsolved_value[1], unsolved_value[2]
    a = (x - inf[0][0]) ** 2 + (y - inf[0][1]) ** 2 + (z - inf[0][2]) ** 2
    b = (x - inf[1][0]) ** 2 + (y - inf[1][1]) ** 2 + (z - inf[1][2]) ** 2
    c = (x - inf[2][0]) ** 2 + (y - inf[2][1]) ** 2 + (z - inf[2][2]) ** 2
    print([x, y, z])
    return [
        math.sqrt(a) - inf[0][3] - (3 * 10 ** 8) * inf[0][4],
        math.sqrt(b) - inf[1][3] - (3 * 10 ** 8) * inf[1][4],
        math.sqrt(c) - inf[2][3] - (3 * 10 ** 8) * inf[2][4]
    ]


# 针对攻击后的数据进行定位
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


def print_revpos1(i):
    get_putongdata(i)
    so = fsolve(get_position, [0, 0, 0])
    print("攻击前接收机位置", so)
    return so


###############################################################################################

# 改变info全局变量
def get_pos(i):
    global info
    info = getdata(i)


# 按单个行获取解算结果，得到接收机位置
def get_position2(unsolved_value):
    x, y, z = unsolved_value[0], unsolved_value[1], unsolved_value[2]
    a = (x - info[0][0]) ** 2 + (y - info[0][1]) ** 2 + (z - info[0][2]) ** 2
    b = (x - info[1][0]) ** 2 + (y - info[1][1]) ** 2 + (z - info[1][2]) ** 2
    c = (x - info[2][0]) ** 2 + (y - info[2][1]) ** 2 + (z - info[2][2]) ** 2
    return [
        math.sqrt(a) - info[0][3] - (3 * 10 ** 8) * info[0][4],
        math.sqrt(b) - info[1][3] - (3 * 10 ** 8) * info[1][4],
        math.sqrt(c) - info[2][3] - (3 * 10 ** 8) * info[2][4]
    ]


##################################################################################################
# 从csv中读取数据，按行号连续取三行进行解算
def get_position3(unsolved_value):
    inf3 = getdata(num)
    x, y, z = unsolved_value[0], unsolved_value[1], unsolved_value[2]
    a = (x - inf3[0][0]) ** 2 + (y - inf3[0][1]) ** 2 + (z - inf3[0][2]) ** 2
    b = (x - inf3[1][0]) ** 2 + (y - inf3[1][1]) ** 2 + (z - inf3[1][2]) ** 2
    c = (x - inf3[2][0]) ** 2 + (y - inf3[2][1]) ** 2 + (z - inf3[2][2]) ** 2
    return [
        math.sqrt(a) - inf3[0][3] - (3 * 10 ** 8) * inf3[0][4],
        math.sqrt(b) - inf3[1][3] - (3 * 10 ** 8) * inf3[1][4],
        math.sqrt(c) - inf3[2][3] - (3 * 10 ** 8) * inf3[2][4]
    ]


#############################################################################################
# 更新攻击后后的数据
def update_pro(data, input):
    n = len(data)
    i = 0
    while i < n:
        data[i][3] = round(input[i], 4)
        i += 1
    return data


# 获取攻击后的数据矩阵
def get_attpreandnext(i):
    data = getdata(i)
    # print(data)
    input = [data[0][3], data[1][3], data[2][3]]
    t = readbit.get_attackpro(input, readbit.get_data(i))
    res = update_pro(data, t)
    print("攻击后数据矩阵：", res)
    return res


# 将获取的攻击矩阵赋值给全局变量inf4
def get_array(i):
    global inf4
    inf4 = get_attpreandnext(i)


# 按输入的数据矩阵解算位置
def get_position4(unsolved_value):
    x, y, z = unsolved_value[0], unsolved_value[1], unsolved_value[2]
    # print(x, y, z)
    a = (x - inf4[0][0]) ** 2 + (y - inf4[0][1]) ** 2 + (z - inf4[0][2]) ** 2
    b = (x - inf4[1][0]) ** 2 + (y - inf4[1][1]) ** 2 + (z - inf4[1][2]) ** 2
    c = (x - inf4[2][0]) ** 2 + (y - inf4[2][1]) ** 2 + (z - inf4[2][2]) ** 2
    return [
        math.sqrt(a) - inf4[0][3] - (2.998 * 10 ** 8) * inf4[0][4],
        math.sqrt(b) - inf4[1][3] - (2.998 * 10 ** 8) * inf4[1][4],
        math.sqrt(c) - inf4[2][3] - (2.998 * 10 ** 8) * inf4[2][4]
    ]


# 获取攻击后解算的接收机位置
def print_revpos2(i):
    get_array(i)
    so = fsolve(get_position4, [0, 0, 0])
    print("攻击后接收机位置：", so)
    return so


################################################################################################

# 获取两个点之间的距离
def get_distance1(pos1, pos2):
    v1 = numpy.array([pos1[0], pos1[1], pos1[2]])
    v2 = numpy.array([pos2[0], pos2[1], pos2[2]])
    distance = numpy.linalg.norm(v1 - v2)
    return distance


# 获取修复钟差的伪距
def get_distance2(pos1, pos2, clk):
    v1 = numpy.array([pos1[0], pos1[1], pos1[2]])
    v2 = numpy.array([pos2[0], pos2[1], pos2[2]])
    distance = numpy.linalg.norm(v1 - v2)
    res = distance - (3 * 10 ** 8) * clk[0]
    return res


################################################################################################
# 获取顺序结算的结果和误差
def get_result():
    solve = fsolve(get_position, [0, 0, 0])
    # 创建一个workbook 设置编码
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建一个worksheet，并给出工作表名（sheet）
    worksheet = workbook.add_sheet('My Worksheet')
    k = 1
    while k < 6825:
        num = k
        so = fsolve(get_position3, [0, 0, 0])
        dis1 = get_distance1(solve, so)
        if dis1 < 500:
            worksheet.write(k, 0, label=num)
            worksheet.write(k, 1, label=so[0])
            worksheet.write(k, 2, label=so[1])
            worksheet.write(k, 3, label=so[2])
            worksheet.write(k, 4, label=dis1)
        k += 1
    workbook.save('result/result1.xls')


###################################################################################################
# 获取测试比较数据
def data_compare(p):
    print("第一组数据解算接收机位置为：")
    solve = fsolve(get_position, [0, 0, 0])
    print(solve)
    print("第二组数据解算接收机位置为：")
    get_pos(p)
    so = fsolve(get_position2, [0, 0, 0])
    print(so)
    print("两次解算的数据差：")
    dis1 = get_distance1(solve, so)
    print(dis1)

    print("攻击后位置：")
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
    clk = [[-0.000239383603], [-0.000243993114], [-0.000172589899], [0.000013685894], [0.000228495598],
           [-0.000324999128]]
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


####################################################################################################
# data_compare(10)
so1 = print_revpos1(10)
so2 = print_revpos2(10)
dis = get_distance1(so1, so2)
print(dis)
