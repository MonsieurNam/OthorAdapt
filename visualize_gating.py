# @title visualize_gating.py
# %%writefile /content/SingloRA_CLIP/visualize_gating.py

import torch
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from loralib.utils import apply_adapter, load_adapter, mark_only_lora_as_trainable

# Cấu hình để Hook bắt dữ liệu
gating_data = []

def get_gating_hook(class_name):
    """
    Tạo một hook function để lưu giá trị Gating weights.
    Sửa đổi để xử lý cấu trúc Sequence của ViT (L, N, H).
    """
    def hook(module, input, output):
        # output shape: [Seq_Len, Batch, Num_Heads]
        # Ví dụ: [50, 1, 2]
        scores = output.detach().cpu().numpy()

        # Xử lý tùy theo shape đầu ra
        if len(scores.shape) == 3:
            # Lấy CLS token (index 0 của dim 0) và ảnh đầu tiên (index 0 của dim 1)
            # Shape kết quả: [Num_Heads]
            current_score = scores[0, 0, :]
        elif len(scores.shape) == 2:
            # Trường hợp batch first hoặc đã pooled: [Batch, Num_Heads]
            current_score = scores[0, :]
        else:
            # Fallback
            current_score = scores.flatten()

        # Lưu giá trị (ép kiểu float để tránh lỗi numpy array)
        gating_data.append({
            'class': class_name,
            'head_0': float(current_score[0]),
            'head_1': float(current_score[1]) if len(current_score) > 1 else 0.0
        })
    return hook

def load_images_from_folder(folder_path, num_imgs=20):
    images = []
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} not found.")
        return []

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # Lấy ngẫu nhiên hoặc lấy đầu danh sách
    files = files[:num_imgs]

    for f in files:
        images.append(os.path.join(folder_path, f))
    return images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='ViT-B/16', help='CLIP backbone')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to OH-SingLoRA checkpoint (.pt)')
    parser.add_argument('--data_root', type=str, required=True, help='Root path to dataset (e.g., /root/DATA/eurosat/2750)')
    parser.add_argument('--class_a', type=str, default='Forest', help='Name of first class folder')
    parser.add_argument('--class_b', type=str, default='Highway', help='Name of second class folder')
    parser.add_argument('--layer_idx', type=int, default=11, help='Layer to analyze gating')
    parser.add_argument('--num_imgs', type=int, default=20, help='Number of images per class to visualize')

    # Các tham số adapter (phải khớp với lúc train)
    parser.add_argument('--r', type=int, default=2)
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=2)

    args = parser.parse_args()

    # 1. Load Model & Adapter
    print("Loading CLIP model...")
    model, preprocess = clip.load(args.backbone, device='cuda')

    # Fake args object để tái sử dụng hàm apply_adapter
    class AdapterArgs:
        adapter = 'ohsinglora'
        encoder = 'both'
        position = 'all'
        params = ['q', 'k', 'v'] # Giả định train full qkv
        r = args.r
        alpha = args.alpha
        ramp_up_steps = 100 # Không quan trọng lúc eval
        num_heads = args.num_heads
        dropout_rate = 0.0
        backbone = args.backbone

    print("Applying OH-SingLoRA Adapter...")
    apply_adapter(AdapterArgs(), model)
    # 1. Chuyển toàn bộ model sang GPU và ép về Float16 (Half)
    model.cuda()
    model.to(model.dtype)

    # 2. FIX LỖI: Tìm tất cả các lớp LayerNorm và ép ngược về Float32
    # CLIP yêu cầu LayerNorm phải chạy ở Float32 để tránh tràn số
    for name, module in model.named_modules():
        if "LayerNorm" in type(module).__name__:
            module.float()
    print(f"Loading weights from {args.checkpoint}...")
    # Tự load tay vì hàm load_adapter trong utils yêu cầu cấu trúc thư mục cứng
    ckpt = torch.load(args.checkpoint, map_location='cuda')
    model.load_state_dict(ckpt['weights'], strict=False)
    model.eval()

    # 2. Register Hook
    # Target: visual.transformer.resblocks[layer].attn.q_proj.gating_network
    # Lưu ý: Cần kiểm tra xem model là 'q_proj' hay 'q_proj.gating_network'
    # Trong code layers_OH_singlora: self.gating_network là một nn.Sequential
    target_layer = model.visual.transformer.resblocks[args.layer_idx].attn.q_proj.gating_network

    # Đăng ký hook
    handle_a = target_layer.register_forward_hook(get_gating_hook(args.class_a))

    # 3. Inference Loop
    print(f"Running inference on {args.class_a} and {args.class_b}...")

    img_paths_a = load_images_from_folder(os.path.join(args.data_root, args.class_a), args.num_imgs)
    img_paths_b = load_images_from_folder(os.path.join(args.data_root, args.class_b), args.num_imgs)

    # Chạy class A
    # Hook đã đăng ký với label Class A
    with torch.no_grad():
        for p in img_paths_a:
            img = preprocess(Image.open(p)).unsqueeze(0).cuda()
            model.encode_image(img)

    # Gỡ hook cũ, đăng ký hook mới cho Class B để đổi label
    handle_a.remove()
    handle_b = target_layer.register_forward_hook(get_gating_hook(args.class_b))

    # Chạy class B
    with torch.no_grad():
        for p in img_paths_b:
            img = preprocess(Image.open(p)).unsqueeze(0).cuda()
            model.encode_image(img)

    handle_b.remove()

    # 4. Visualization
    print("Plotting Gating Map...")

    # Chuẩn bị dữ liệu cho Heatmap
    # Tạo matrix: Rows = Images (20 Forest + 20 Highway), Cols = Head 0, Head 1
    data_matrix = np.zeros((len(gating_data), 2))
    labels = []

    for i, item in enumerate(gating_data):
        data_matrix[i, 0] = item['head_0']
        data_matrix[i, 1] = item['head_1']
        labels.append(f"{item['class']} {i%args.num_imgs + 1}")

    plt.figure(figsize=(8, 10))
    sns.heatmap(data_matrix, annot=True, cmap="YlGnBu", yticklabels=labels, xticklabels=['Head 1', 'Head 2'])

    plt.title(f'Gating Activation Map (Layer {args.layer_idx})\n{args.class_a} vs {args.class_b}')
    plt.tight_layout()

    save_path = 'gating_map_visualization.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved visualization to {save_path}")

    # Vẽ biểu đồ Bar trung bình để thấy rõ sự chuyên biệt
    plt.figure(figsize=(6, 5))

    avg_a_h0 = np.mean([d['head_0'] for d in gating_data if d['class'] == args.class_a])
    avg_a_h1 = np.mean([d['head_1'] for d in gating_data if d['class'] == args.class_a])

    avg_b_h0 = np.mean([d['head_0'] for d in gating_data if d['class'] == args.class_b])
    avg_b_h1 = np.mean([d['head_1'] for d in gating_data if d['class'] == args.class_b])

    x = np.arange(2)
    width = 0.35

    plt.bar(x - width/2, [avg_a_h0, avg_b_h0], width, label='Head 1', color='skyblue')
    plt.bar(x + width/2, [avg_a_h1, avg_b_h1], width, label='Head 2', color='orange')

    plt.xticks(x, [args.class_a, args.class_b])
    plt.ylabel('Average Gating Weight')
    plt.title('Head Specialization Analysis')
    plt.legend()
    plt.savefig('head_specialization.png', dpi=300)
    print("Saved specialization chart to head_specialization.png")

if __name__ == "__main__":
    main()
    
# python3 visualize_gating.py \
#   --checkpoint "/content/drive/MyDrive/RESEARCH/OHSinglora_CLIP/checkpoints_rank2/ohsinglora/ViT-B16/eurosat/4shots/seed1/adapter_weights.pt" \
#   --data_root "/content/DATA/eurosat/2750" \
#   --class_a "Forest" \
#   --class_b "Highway" \
#   --num_imgs 20 \
#   --layer_idx 11