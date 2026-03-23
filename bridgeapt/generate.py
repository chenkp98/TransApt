import torch
import torch.nn.functional as F

from .model import BridgeAPT, GVP

NUCLEOTIDES = ['X', 'A', 'T', 'C', 'G']


def generate_sequence(model, coords, scalar_features, temperature=1.0):
    """
    对给定坐标和标量特征进行推理，返回核酸序列字符串列表。

    Args:
        model: BridgeAPT 模型实例
        coords: [batch, length, 21] float32 张量
        scalar_features: [batch, length, 6] float32 张量
        temperature: 采样温度，默认 1.0

    Returns:
        list[str]: 生成的核酸序列列表
    """
    with torch.no_grad():
        logits = model(coords, scalar_features)
        probs = F.softmax(logits / temperature, dim=-1)
        samples = torch.multinomial(probs.view(-1, 5), 1).view(coords.shape[0], -1)
        sequences = []
        for batch in range(samples.shape[0]):
            seq = ''.join([NUCLEOTIDES[i] for i in samples[batch].tolist()])
            sequences.append(seq)
        return sequences
