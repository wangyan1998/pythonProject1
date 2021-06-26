# latitude:纬度 longitude:经度 altitude:海拔
import math
import numpy


def LLA_to_XYZ(latitude, longitude, altitude):
    # 经纬度的余弦值
    cosLat = math.cos(latitude * math.pi / 180)
    sinLat = math.sin(latitude * math.pi / 180)
    cosLon = math.cos(longitude * math.pi / 180)
    sinLon = math.sin(longitude * math.pi / 180)

    # WGS84坐标系的参数
    rad = 6378137.0  # 地球赤道平均半径（椭球长半轴：a）
    f = 1.0 / 298.257224  # WGS84椭球扁率 :f = (a-b)/a
    C = 1.0 / math.sqrt(cosLat * cosLat + (1 - f) * (1 - f) * sinLat * sinLat)
    S = (1 - f) * (1 - f) * C
    h = altitude

    # 计算XYZ坐标
    X = (rad * C + h) * cosLat * cosLon
    Y = (rad * C + h) * cosLat * sinLon
    Z = (rad * S + h) * sinLat

    return numpy.array([X, Y, Z])


def XYZ_to_LLA(X, Y, Z):
    # WGS84坐标系的参数
    a = 6378137.0  # 椭球长半轴
    b = 6356752.314245  # 椭球短半轴
    ea = numpy.sqrt((a ** 2 - b ** 2) / a ** 2)
    eb = numpy.sqrt((a ** 2 - b ** 2) / b ** 2)
    p = numpy.sqrt(X ** 2 + Y ** 2)
    theta = numpy.arctan2(Z * a, p * b)

    # 计算经纬度及海拔
    longitude = numpy.arctan2(Y, X)
    latitude = numpy.arctan2(Z + eb ** 2 * b * numpy.sin(theta) ** 3, p - ea ** 2 * a * numpy.cos(theta) ** 3)
    N = a / numpy.sqrt(1 - ea ** 2 * numpy.sin(latitude) ** 2)
    altitude = p / numpy.cos(latitude) - N

    return numpy.array([numpy.degrees(latitude), numpy.degrees(longitude), altitude])


def get_distance1(position1, position2, clk):
    v1 = LLA_to_XYZ(position1[0], position1[1], position1[2])
    v2 = LLA_to_XYZ(position2[0], position2[1], position2[2])
    distance = numpy.linalg.norm(v1 - v2)
    res = distance + (3 * 10 ** 8) * clk[0]

    return res


def get_distance2(pos1, pos2, clk):
    v1 = numpy.array([pos1[0], pos1[1], pos1[2]])
    v2 = numpy.array([pos2[0], pos2[1], pos2[2]])
    distance = numpy.linalg.norm(v1 - v2)
    res = distance - (3 * 10 ** 8) * clk[0]

    return res


solved = get_distance1([-1.11933, 90.79556, 20164300.4], [30.51556, 114.49100, 74.3385], [0.000104647296])
solved1 = get_distance2([-2279829.2223, 5004708.9330, 3219779.0345], [-368461.739, 26534822.568, -517664.322],
                        [-0.000104647296])
# pos = [-2279772.39208934,  5004740.35324707,  3219737.89319037]
pos = [-106650.20098951, 5549197.37127125, 3139363.97226811]
solved2 = XYZ_to_LLA(pos[0], pos[1], pos[2])
print(solved)
print(solved1)
print(solved2)
