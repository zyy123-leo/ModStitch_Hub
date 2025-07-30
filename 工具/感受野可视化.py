import matplotlib.pyplot as plt
import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_network(weight_path):
    # ====== 把你的网络 import 进来 ====== #
    # 举例 from nets.segformer import SegFormer
    # net = SegFormer(num_classes=2,phi='b2')
    # =================================== #
    net.load_state_dict(torch.load(weight_path, map_location=device)) \
          if weight_path.endswith('.pth') else torch.load(weight_path, map_location='cpu')
    net.eval()
    return net

class Args:
    def __init__(self):
        self.model = r'1.pth'  # 权重路径
        self.img = '2.jpg'                    # 测试图路径


args = Args()

# 加载模型
model = load_network(args.model).to(device)

input_tensor = torch.zeros((1, 3, 512, 512)).to(device)
center = (512 // 2, 512 // 2)
input_tensor[0, :, center[0], center[1]] = 1.0
# img_tensor = input_tensor.unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    output = model(input_tensor)
feature_maps = output.squeeze().cpu().numpy()
# feature_maps = feature_maps / np.max(feature_maps)  # 归一化
layer_name = 'layer'
save_dir = 'erf'
# Visualize receptive field
plt.figure(figsize=(10, 10))
plt.imshow(np.mean(feature_maps, axis=0), cmap='viridis')
plt.colorbar()
plt.title(f'Effective Receptive Field - {layer_name}')
plt.savefig(f'{layer_name}_receptive_field1.png')
plt.close()
