import numpy


def My_Loss(input, target):
    v1 = numpy.array([input[0], input[1], input[2]])
    v2 = numpy.array([target[0], target[1], target[2]])
    distance = numpy.linalg.norm(v1 - v2)
    return distance