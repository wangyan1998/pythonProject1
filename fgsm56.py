from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

pretrained_model = "./mymodel/model11.pth"
use_cuda = True


# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.7)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # 定义前向传播网络
    def forward(self, x):
        print(x.shape)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        print(x.shape)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        print(x.shape)
        x = x.view(-1, 320)
        print(x.shape)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = F.dropout(x, training=self.training)
        print(x.shape)
        x = self.fc2(x)
        print(x.shape)
        return F.log_softmax(x, dim=1)


# Define what device we are using
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)
print(model)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

data = torch.rand(1, 1, 28, 28)
target = torch.tensor([5])


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    # 得到数据梯度的符号
    # print(image.shape)
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    # 通过梯度符号获得新的处理过的图片
    perturbed_image = image + epsilon * sign_data_grad
    # print(perturbed_image)
    # Adding clipping to maintain [0,1] range
    # 将张量压缩至[0,1]之间
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # print(perturbed_image)
    # Return the perturbed image
    return perturbed_image


def test(model, device, data, target, epsilon):
    # Send the data and label to the device
    # 将数据和标签发送到设备
    data, target = data.to(device), target.to(device)

    # Set requires_grad attribute of tensor. Important for Attack
    # 集合需要张量的梯度属性。对攻击很重要
    data.requires_grad = True

    # Forward pass the data through the model
    # 通过模型前向传播数据
    output = model(data)
    print("output:")
    print(output)
    print(output)
    print("target:")
    print(target)

    init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
    print(init_pred)

    # If the initial prediction is wrong, dont bother attacking, just move on

    # Calculate the loss
    loss = F.nll_loss(output, target)
    # loss=output[0][target[0]]
    print(loss)
    # print(loss.shape)

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = data.grad.data

    # Call FGSM Attack
    perturbed_data = fgsm_attack(data, epsilon, data_grad)
    # print(perturbed_data[0])

    # Re-classify the perturbed image
    output = model(perturbed_data)
    print(output)
    final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
    print(final_pred)
    # Check for success


    # Return the accuracy and an adversarial example
    return 0, 0


accuracies = []
examples = []
acc, ex = test(model, device, data, target, 0.05)
