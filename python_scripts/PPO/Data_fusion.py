import torch
from python_scripts.Project_config import device

def data_fusion(image_features, robot_angles, gnn_features):
    """数据融合函数,将图像特征、机器人状态和GNN特征融合
    参数:
        image_features: 经过预处理的图像特征向量 
        robot_angles: 机器人舵机角度数据
        gnn_features: 图神经网络输出的特征向量
    返回:
        融合后的特征向量
    """
    # 将所有特征转换为tensor类型
    if not isinstance(image_features, torch.Tensor):
        image_features = torch.tensor(image_features, dtype=torch.float32)
    if not isinstance(robot_angles, torch.Tensor):
        robot_angles = torch.tensor(robot_angles, dtype=torch.float32)
    if not isinstance(gnn_features, torch.Tensor):
        gnn_features = torch.tensor(gnn_features, dtype=torch.float32)
        
    # 对每个特征进行归一化处理
    image_features = torch.nn.functional.normalize(image_features, dim=0)
    robot_angles = torch.nn.functional.normalize(robot_angles, dim=0) 
    gnn_features = torch.nn.functional.normalize(gnn_features, dim=0)
    
    # 拼接所有特征
    fused_features = torch.cat([image_features, robot_angles, gnn_features], dim=0)
    
    # 使用全连接层进行特征融合
    fusion_layer = torch.nn.Linear(fused_features.shape[0], 512).to(device)
    fused_features = fusion_layer(fused_features)
    fused_features = torch.nn.functional.relu(fused_features)
    
    return fused_features