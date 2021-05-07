import numpy
import torch

# 输出 pytorch 版本
# print("torch version {}".format(torch.__version__))
# print("cuda is available {}".format(torch.cuda.is_available()))

# 生成张量
# data = [[1, 2], [3, 4]]
# x_data = torch.tensor(data)
# x_ones = torch.ones_like(x_data)  # 保留 x_data 的属性
# print(f"Ones Tensor: \n {x_ones} \n")
#
# x_rand = torch.rand_like(x_data, dtype=torch.float)  # 重写 x_data 的数据类型int -> float
# print(f"Random Tensor: \n {x_rand} \n")

# 设定数据维度，生成张量
# shape = (2, 3,)
# rand_tensor = torch.rand(shape)
# ones_tensor = torch.ones(shape)
# zeros_tensor = torch.zeros(shape)
#
# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")

# 输出张量的属性
# tensor = torch.rand(3, 4)
# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")

# # 判断当前环境GPU是否可用, 然后将tensor导入GPU内运行
# tensor = torch.rand(3, 4)
# if torch.cuda.is_available():
#     tensor = tensor.to('cuda')
#
# tensor = torch.ones(4, 4)
# tensor[:, 1] = 0  # 将第1列(从0开始)的数据全部赋值为0
# print(tensor)

# 拼接张量
# tensor = torch.ones(4, 4)
# t1 = torch.cat([tensor, tensor, tensor], dim=1)
# print(t1)

# # 逐个元素相乘结果
# print(f"tensor.mul(tensor): \n {tensor.mul(tensor)} \n")
# # 等价写法:
# print(f"tensor * tensor: \n {tensor * tensor}")
#
# print(f"tensor.matmul(tensor.T): \n {tensor.matmul(tensor.T)} \n")
# # 等价写法:
# print(f"tensor @ tensor.T: \n {tensor @ tensor.T}")

# 张量运算
# print(tensor, "\n")
# tensor.add_(5)
# print(tensor)

# 张量换成numpy的array
# t = torch.ones(5)
# print(f"t: {t}")
# n = t.numpy()
# print(f"n: {n}")
# t.add_(1)
# print(f"t: {t}")
# print(f"n: {n}")

# numpy的array换成张量
# n = numpy.ones(5)
# t = torch.from_numpy(n)
# print(f"t: {t}")
# print(f"n: {n}")
# numpy.add(n, 1, out=n)
# print(f"t: {t}")
# print(f"n: {n}")
a = torch.randn(4, 4)
print(a)
print(a.size())
b = a.view(2, 8)
print(b)
