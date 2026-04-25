# @title visualize_combined.py
import numpy as np
import matplotlib.pyplot as plt

def plot_combined_chart():
    # Cấu hình font và style học thuật
    plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})
    fig = plt.figure(figsize=(16, 7.5)) # Kích thước rộng để chứa 2 subplots

    # ==========================================
    # 1. DỮ LIỆU TỪ TABLE 2 (4-SHOT)
    # ==========================================
    datasets =['Aircraft', 'EuroSAT', 'Food101', 'OxfordPets', 'Flowers102', 'Caltech101', 'DTD', 'UCF101']
    
    methods_data = {
        'CoOp (M=4)':[29.70, 65.80, 83.50, 92.30, 86.60, 94.50, 58.50, 78.10],
        'CoOp (M=16)':[30.90, 69.70, 84.50, 92.50, 92.20, 94.50, 59.50, 77.60],
        'CoCoOp':[30.60, 61.70, 86.30, 92.70, 81.50, 94.80, 55.70, 75.30],
        'CLIP-Adapter':[27.90, 51.20, 86.50, 90.80, 73.10, 94.00, 46.10, 70.60],
        'Tip-Adapter-F':[35.70, 76.80, 86.50, 91.90, 92.10, 94.80, 59.80, 78.10],
        'PLOT++':[35.30, 83.20, 86.50, 92.60, 92.90, 95.10, 62.40, 79.80],
        'KgCoOp':[32.20, 71.80, 86.90, 92.60, 87.00, 95.00, 58.70, 77.60],
        'TaskRes':[33.40, 74.20, 86.00, 91.90, 85.00, 95.00, 60.10, 76.20],
        'MaPLe':[30.10, 69.90, 86.70, 93.30, 84.90, 95.00, 59.00, 77.10],
        'ProGrad':[34.10, 69.60, 85.40, 92.10, 91.10, 94.40, 59.70, 77.90],
        'CLIP-LoRA':[39.15, 87.49, 83.72, 91.06, 93.67, 95.58, 65.19, 80.39],
        'OrthoAdapt (Ours)':[38.61, 89.06, 84.30, 92.18, 93.30, 95.62, 65.72, 80.86]
    }

    # ==========================================
    # SUBPLOT 1: RADAR CHART (Trái)
    # ==========================================
    ax1 = fig.add_subplot(121, polar=True)
    N = len(datasets)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1] # Khép kín vòng
    
    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    plt.xticks(angles[:-1], datasets, size=11, fontweight='bold')
    ax1.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100],["20", "40", "60", "80", "100"], color="grey", size=9)
    plt.ylim(20, 100)

    # List màu pastel (mờ) cho các phương pháp cũ
    base_colors = plt.cm.tab20.colors
    
    for idx, (name, values) in enumerate(methods_data.items()):
        val_closed = values + values[:1] # Khép kín vòng
        
        if name == 'OrthoAdapt (Ours)':
            ax1.plot(angles, val_closed, linewidth=2.5, linestyle='solid', color='#1f77b4', marker='s', markersize=6, label=name, zorder=11)
            ax1.fill(angles, val_closed, color='#1f77b4', alpha=0.15, zorder=11)
        elif name == 'CLIP-LoRA':
            ax1.plot(angles, val_closed, linewidth=2.5, linestyle='solid', color='#d62728', marker='o', markersize=6, label=name, zorder=10)
        else:
            # Các phương pháp cũ: Nét đứt mảnh, màu mờ để làm nền
            ax1.plot(angles, val_closed, linewidth=1, linestyle='--', color=base_colors[idx%10], alpha=0.6, label=name, zorder=5)

    ax1.grid(color='#E8E8E8', linestyle='-', linewidth=1)
    ax1.spines['polar'].set_color('#E8E8E8')
    ax1.set_title("(a) Performance Overview vs. All Methods", size=14, fontweight='bold', pad=30)
    
    # Legend cho Radar chart (đặt ở dưới cùng, chia làm 4 cột)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=9.5, frameon=False)


    # ==========================================
    # SUBPLOT 2: DELTA BAR CHART (Phải)
    # ==========================================
    # ==========================================
    # SUBPLOT 2: DELTA BAR CHART (Phải)
    # ==========================================
    ax2 = fig.add_subplot(122)
    
    clip_lora = np.array(methods_data['CLIP-LoRA'])
    orthoadapt = np.array(methods_data['OrthoAdapt (Ours)'])
    deltas = orthoadapt - clip_lora

    sorted_indices = np.argsort(deltas)[::-1]
    datasets_sorted = [datasets[i] for i in sorted_indices]
    deltas_sorted = deltas[sorted_indices]

    colors =['#1f77b4' if val > 0 else '#d62728' for val in deltas_sorted]
    y_pos = np.arange(len(datasets_sorted))
    
    bars = ax2.barh(y_pos, deltas_sorted, color=colors, edgecolor='black', height=0.6, alpha=0.85)

    ax2.axvline(0, color='black', linewidth=1.5, linestyle='--')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(datasets_sorted, fontweight='bold')
    ax2.invert_yaxis()

    # --- FIX OVERLAY HERE ---
    # Lấy min và max của trục x, sau đó cộng/trừ thêm padding để có đủ khoảng trống cho text
    min_val = np.min(deltas_sorted)
    max_val = np.max(deltas_sorted)
    ax2.set_xlim(min_val - 1.0, max_val + 1.0) 
    # ------------------------

    ax2.set_xlabel('Absolute Accuracy Gain (%) over CLIP-LoRA', fontweight='bold', size=12)
    ax2.set_title('(b) OrthoAdapt vs. CLIP-LoRA', size=14, fontweight='bold', pad=20)
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    for bar, val in zip(bars, deltas_sorted):
        # Tăng offset lên 0.1 để chữ không bị dính sát vào bar
        x_pos = val + 0.1 if val > 0 else val - 0.1
        ha = 'left' if val > 0 else 'right'
        sign = "+" if val > 0 else ""
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2, f'{sign}{val:.2f}%', 
                 va='center', ha=ha, fontsize=11, fontweight='bold', color=bar.get_facecolor())

    # Đẩy chú thích Baseline vào bên trong một chút để không che đồ thị
    ax2.text(0.05, len(datasets_sorted)-0.2, 'Baseline: CLIP-LoRA (0%)', 
             fontsize=11, color='black', style='italic', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax2.text(0.05, len(datasets_sorted)-0.2, 'Baseline: CLIP-LoRA (0%)', 
             fontsize=11, color='black', style='italic', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # ==========================================
    # LƯU VÀ XUẤT ẢNH
    # ==========================================
    plt.tight_layout()
    # Chỉnh khoảng cách giữa 2 đồ thị
    plt.subplots_adjust(wspace=0.3, bottom=0.25) 
    
    plt.savefig('combined_performance_4shot.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('combined_performance_4shot.png', format='png', dpi=300, bbox_inches='tight')
    print("✅ Combined chart generated successfully: 'combined_performance_4shot.pdf' and 'combined_performance_4shot.png'")

if __name__ == '__main__':
    plot_combined_chart()