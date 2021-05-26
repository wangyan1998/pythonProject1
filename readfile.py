import csv
import pandas as pd
import numpy as np
import xlsxwriter
import xlwt


# workbook = xlsxwriter.Workbook('D:\\Desktop\\result.xlsx')
# worksheet = workbook.add_worksheet()


def getdata(i):
    result = openfile()
    k = i
    j = i + 3
    res = []
    while k < j:
        a = [float(result[k][2]), float(result[k][3]), float(result[k][4]), float(result[k][5]), float(result[k][6])]
        res.append(a)
        k += 1
    return res


def openfile():
    f = open('dataset/mydataset.csv', 'r')
    reader = csv.reader(f)  # 创建一个与该文件相关的阅读器
    result = list(reader)
    return result


getdata(1)
