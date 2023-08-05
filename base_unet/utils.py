import torch


def corrupt(x, amount):
    """
    添加噪声
    :param x: 输入
    :param amount: 控制噪声量，amount=1 为全是噪声，amount=0 没有噪声， 0<=amount<=1
    :return:
    """
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)
    return x * (1 - amount) + noise * amount
