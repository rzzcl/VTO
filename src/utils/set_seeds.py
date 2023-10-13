import random
import os
import numpy as np
import torch
import accelerate


def set_seed(seed):
    """
    Set seed for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    accelerate.utils.set_seed(seed)


# 在深度学习任务中，随机性是常见的，如参数初始化、数据加载、优化算法等。通过设置相同的随机种子，可以确保在每次运行代码时生成的随机数序列是一样的，从而使结果可重复。

# 函数 set_seed(seed) 的具体步骤如下：

# 使用 random.seed(seed) 设置 Python 内置的随机数生成器的种子。
# 使用 os.environ['PYTHONHASHSEED'] = str(seed) 设置一个环境变量，以确保散列函数的输出也是可重复的。
# 使用 np.random.seed(seed) 设置 NumPy 库中的随机数生成器的种子。
# 使用 torch.manual_seed(seed) 设置 Torch 库中的随机数生成器的种子。
# 使用 torch.cuda.manual_seed(seed) 设置 CUDA 设备上的随机数生成器的种子。
# 使用 torch.backends.cudnn.deterministic = True 设置 PyTorch 的 cuDNN 加速库为确定性模式，以进一步增加结果的可重复性。
# 使用 accelerate.utils.set_seed(seed)（假设存在 accelerate 库）设置加速器库中的随机种子，以确保在分布式训练或使用加速器的情况下仍能实现结果的可复现性。
# 通过以上步骤，函数 set_seed(seed) 可以将随机性控制在可控的范围内，从而实现结果的可复现性。这对于调试代码、比较不同模型或算法的性能以及确保实验结果一致性非常有用。