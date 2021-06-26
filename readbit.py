import csv

import numpy
import torch


# 打开文件
def open_file():
    f = open('dataset/bitdataset.csv', 'r')
    reader = csv.reader(f)  # 创建一个与该文件相关的阅读器
    result = list(reader)
    return result


# 获取bit位矩阵
def get_data(i):
    result = open_file()
    k = i
    j = i + 3
    res = []
    while k < j:
        a = [int(result[k][3]), int(result[k][4]), int(result[k][5]), int(result[k][6]), int(result[k][7]),
             int(result[k][8]), int(result[k][9]),
             int(result[k][10])]
        res.append(a)
        # print(a)
        k += 1
    print(numpy.array(res))
    return res


# 从文件中获取伪距
def get_pro(i):
    result = open_file()
    k = i
    j = i + 3
    res = []
    while k < j:
        res.append(float(result[k][2]))
        k += 1
    # print(res)
    return res


# 攻击矩阵的生成和相加
def attack(input):
    att = torch.rand(3, 8)
    a = numpy.array(att)
    print(att)
    # print(a)
    a = numpy.int64(a > 0.5)
    # print(a)
    a[0][0] = 0
    a[1][0] = 0
    a[2][0] = 0
    print(a)
    res = input + a
    print(res)
    return res


# 计算bit位的和
def compute(arr):
    data = arr[0] * 1000 + arr[1] * 100 + arr[2] * 10 + arr[3] + arr[4] * 0.1 + arr[5] * 0.01 + arr[6] * 0.001 + arr[
        7] * 0.0001
    # print(round(data, 4))
    return round(data, 4)


# 计算伪距后8位攻击前后bit位计算的和
def attack_pro(data):
    i = 0
    j = len(data)
    res = []
    result = []
    while i < j:
        k = compute(data[i])
        res.append(k)
        i += 1
    # print(res)
    result.append(res)
    data1 = attack(data)
    res1 = []
    i = 0
    while i < j:
        k = compute(data1[i])
        res1.append(k)
        i += 1
    result.append(res1)
    return result


# 计算攻击前后的伪距
def get_attackpro(input, data):
    print("攻击前的伪距值：")
    print(input)
    i = 0
    j = len(data)
    res = []
    result = []
    while i < j:
        k = compute(data[i])
        input[i] -= k
        res.append(k)
        i += 1
    # print(res)
    result.append(res)
    data1 = attack(data)
    res1 = []
    i = 0
    while i < j:
        k = compute(data1[i])
        input[i] += k
        res1.append(k)
        i += 1
    print("攻击后的伪距值：")
    print(input)
    result.append(res1)
    return input


# d = get_data(7)
# pro = get_pro(7)
# get_attackpro(pro, d)

# attacked = attack(input)
# compute(attacked[0])
# print(attack_pro(input))
