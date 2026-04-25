import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def find_layer_weights_lora(weights_dict, layer_idx, param_name='q_proj'):
    """
    Tìm trọng số A và B trong file checkpoint của LoRA tiêu chuẩn.
    Cấu trúc mong đợi: {'weights': {'layer_X': {'q_proj': {'w_lora_A': ..., 'w_lora_B': ...}}}}
    """
    layer_key = f'layer_{layer_idx}'
    
    if 'weights' in weights_dict:
        weights_dict = weights_dict['weights']
    
    if layer_key not in weights_dict:
        print(f"[LoRA] Warning: Không tìm thấy {layer_key}. Các keys có sẵn: {list(weights_dict.keys())[:5]}")
        return None

    if param_name not in weights_dict[layer_key]:
        print(f"[LoRA] Warning: Không tìm thấy tham số {param_name} trong {layer_key}.")
        return None

    w_a = weights_dict[layer_key][param_name]['w_lora_A']
    w_b = weights_dict[layer_key][param_name]['w_lora_B']
    
    return w_a, w_b

def find_layer_weights_oh_singlora(state_dict, layer_idx, param_name='q_proj'):
    """
    Tìm trọng số lora_A_heads trong file checkpoint của OH-SingLoRA.
    Cấu trúc mong đợi: Flat state_dict với key dạng '...resblocks.X.attn.q_proj.lora_A_heads'
    """
    if 'weights' in state_dict:
        state_dict = state_dict['weights']

    # Key tìm kiếm (dựa trên cấu trúc CLIP ViT)
    # Target: visual.transformer.resblocks.11.attn.q_proj.lora_A_heads
    # Hoặc: transformer.resblocks.11.attn.q_proj.lora_A_heads
    search_suffix = f"resblocks.{layer_idx}.attn.{param_name}.lora_A_heads"
    
    target_key = None
    for k in state_dict.keys():
        if k.endswith(search_suffix):
            target_key = k
            break
    
    if target_key is None:
        print(f"[OH-SingLoRA] Warning: Không tìm thấy key chứa '{search_suffix}'.")
        return None
        
    return state_dict[target_key]

def compute_delta_w_lora(w_a, w_b):
    """
    Tính Delta W cho LoRA: B @ A
    """
    delta_w = torch.matmul(w_b, w_a)
    return delta_w

def compute_delta_w_oh_singlora(lora_A_heads):
    """
    Tính Delta W đại diện cho OH-SingLoRA.
    Do có Gating động, ta tính tổng khả năng biểu diễn của các heads:
    Delta_Proxy = Sum(Head_out @ Head_in.T)
    """
    # lora_A_heads_out = lora_A_heads[:, :self.out_features, :]
    # lora_A_heads_in = lora_A_heads[:, :self.in_features, :]
    
    num_heads, d, r = lora_A_heads.shape
    
    out_dim = d
    in_dim = d
    
    lora_A_heads_out = lora_A_heads[:, :out_dim, :]
    lora_A_heads_in = lora_A_heads[:, :in_dim, :]
    
    delta_w_sum = torch.zeros(out_dim, in_dim, device=lora_A_heads.device)
    
    for i in range(num_heads):
        # A_out @ A_in.T
        head_update = torch.matmul(lora_A_heads_out[i], lora_A_heads_in[i].transpose(0, 1))
        delta_w_sum += head_update
        
    return delta_w_sum

def get_singular_values(matrix):
    """
    Tính SVD và trả về các giá trị kỳ dị đã chuẩn hóa.
    """
    matrix_np = matrix.detach().cpu().float().numpy()
    # SVD: U, S, Vh
    _, S, _ = np.linalg.svd(matrix_np, full_matrices=False)
    
    if S[0] > 0:
        S_norm = S / S[0]
    else:
        S_norm = S
    return S_norm

def main():
    parser = argparse.ArgumentParser(description="Spectral Analysis for CLIP Adapters")
    parser.add_argument('--lora_path', type=str, default='', help='Path to LoRA checkpoint (.pt)')
    parser.add_argument('--oh_path', type=str, default='', help='Path to OH-SingLoRA checkpoint (.pt)')
    parser.add_argument('--layer_idx', type=int, default=11, help='Transformer layer index to analyze (0-11 for ViT-B)')
    parser.add_argument('--param', type=str, default='q_proj', choices=['q_proj', 'v_proj'], help='Parameter to analyze')
    parser.add_argument('--top_k', type=int, default=50, help='Number of singular values to plot')
    
    args = parser.parse_args()

    if not args.lora_path and not args.oh_path:
        print("Warning: Không có đường dẫn file. Đang chạy chế độ DEMO với dữ liệu giả lập...")
        s_lora = np.exp(-np.arange(args.top_k) * 0.5) 
        s_oh = np.exp(-np.arange(args.top_k) * 0.1)
    else:
        print(f"Loading LoRA: {args.lora_path}")
        lora_ckpt = torch.load(args.lora_path, map_location='cpu')
        w_a, w_b = find_layer_weights_lora(lora_ckpt, args.layer_idx, args.param)
        delta_lora = compute_delta_w_lora(w_a, w_b)
        s_lora = get_singular_values(delta_lora)

        print(f"Loading OH-SingLoRA: {args.oh_path}")
        oh_ckpt = torch.load(args.oh_path, map_location='cpu')
        lora_heads = find_layer_weights_oh_singlora(oh_ckpt, args.layer_idx, args.param)
        delta_oh = compute_delta_w_oh_singlora(lora_heads)
        s_oh = get_singular_values(delta_oh)

    plt.figure(figsize=(10, 6))
    
    x_axis = np.arange(min(args.top_k, len(s_lora)))
    
    plt.plot(x_axis, s_lora[:len(x_axis)], 'o--', color='#d62728', label='Standard LoRA', linewidth=2, markersize=5, alpha=0.8)
    plt.plot(x_axis, s_oh[:len(x_axis)], 's-', color='#1f77b4', label='OH-SingLoRA (Ours)', linewidth=2, markersize=5)

    plt.title(f'Singular Value Spectrum Analysis\nLayer {args.layer_idx} - {args.param}', fontsize=14, fontweight='bold')
    plt.xlabel('Singular Value Index (Sorted)', fontsize=12)
    plt.ylabel('Normalized Singular Value (Log Scale)', fontsize=12)
    plt.yscale('log') 
    
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12)
    
    plt.text(x_axis[len(x_axis)//2], 0.1, "Steeper drop = Rank Collapse\nFlatter = Richer Representation", 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    save_path = 'singular_value_spectrum.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] Biểu đồ đã được lưu tại: {save_path}")
    print("Hãy mở ảnh này lên. Nếu đường màu Xanh (OH-SingLoRA) nằm trên đường màu Đỏ (LoRA), chúng ta đã thắng!")

if __name__ == '__main__':
    main()
    
# python3 analyze_spectrum.py \
#   --lora_path "./checkpoints/lora/vit-b16/eurosat/16shots/seed1/lora_weights.pt" \
#   --oh_path "./checkpoints/ohsinglora/vit-b16/eurosat/16shots/seed1/adapter_weights.pt" \
#   --layer_idx 11 & 6 \
#   --param q_proj