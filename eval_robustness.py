# @title eval_robustness.py
# %%writefile /content/SingloRA_CLIP/eval_robustness.py

import torch
import torch.nn as nn
import clip
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import build_dataset
from datasets.utils import DatasetWrapper
from utils import cls_acc
from loralib.utils import apply_adapter, apply_lora
import argparse
import os

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

def get_corrupted_transform(n_px, severity=1):
    base_transforms = [
        T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(n_px),
        T.ToTensor(),
    ]
    if severity == 1:
        base_transforms.append(AddGaussianNoise(std=0.05))
    elif severity == 2:
        base_transforms.append(T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))
        base_transforms.append(AddGaussianNoise(std=0.1))
    elif severity == 3:
        base_transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5))
        base_transforms.append(T.GaussianBlur(kernel_size=5, sigma=(1.0, 3.0)))
        base_transforms.append(AddGaussianNoise(std=0.15))
            
    base_transforms.append(T.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                       (0.26862954, 0.26130258, 0.27577711)))
    return T.Compose(base_transforms)

def evaluate(model, loader, dataset):
    model.eval()
    with torch.no_grad():
        template = dataset.template[0]
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        texts = clip.tokenize(texts).cuda()
        class_embeddings = model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    
    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ class_embeddings.t()
            acc += cls_acc(similarity, target) * len(similarity)
            tot_samples += len(similarity)
            
    return acc / tot_samples

def load_weights_smart(model, checkpoint_path, adapter_type, args):
    print(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Checkpoint file not found!")
        
    ckpt = torch.load(checkpoint_path, map_location='cuda')
    weights = ckpt['weights'] if 'weights' in ckpt else ckpt

    if adapter_type == 'lora':
        print(">> Loading Standard LoRA (Sequential Mapping)...")
        from loralib.utils import INDEX_POSITIONS_VISION, INDEX_POSITIONS_TEXT
        
        global_layer_cnt = 0
        
        if args.encoder in ['text', 'both']:
            indices = INDEX_POSITIONS_TEXT[args.position]
            for i, block in enumerate(model.transformer.resblocks):
                if i in indices:
                    layer_key = f'layer_{global_layer_cnt}'
                    if layer_key in weights:
                        w_dict = weights[layer_key]
                        # Load params
                        for param_char, module in zip(['q', 'k', 'v', 'o'], 
                                                    [block.attn.q_proj, block.attn.k_proj, block.attn.v_proj, block.attn.proj]):
                            if f'{param_char}_proj' in w_dict:
                                module.w_lora_A.data.copy_(w_dict[f'{param_char}_proj']['w_lora_A'])
                                module.w_lora_B.data.copy_(w_dict[f'{param_char}_proj']['w_lora_B'])
                        # print(f"Loaded Text Layer {i} from {layer_key}")
                    global_layer_cnt += 1

        if args.encoder in ['vision', 'both']:
            indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
            for i, block in enumerate(model.visual.transformer.resblocks):
                if i in indices:
                    layer_key = f'layer_{global_layer_cnt}'
                    if layer_key in weights:
                        w_dict = weights[layer_key]
                        for param_char, module in zip(['q', 'k', 'v', 'o'], 
                                                    [block.attn.q_proj, block.attn.k_proj, block.attn.v_proj, block.attn.proj]):
                            if f'{param_char}_proj' in w_dict:
                                module.w_lora_A.data.copy_(w_dict[f'{param_char}_proj']['w_lora_A'])
                                module.w_lora_B.data.copy_(w_dict[f'{param_char}_proj']['w_lora_B'])
                        # print(f"Loaded Vision Layer {i} from {layer_key}")
                    else:
                        print(f"Warning: Key {layer_key} missing for Vision Block {i}")
                    global_layer_cnt += 1
                    
        print(f"Standard LoRA weights loaded successfully. Total layers processed: {global_layer_cnt}")

    else:
        print(">> Loading SingLoRA/OH-SingLoRA (Flat dict)...")
        new_state_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '')
            new_state_dict[new_k] = v
            
        incompatible = model.load_state_dict(new_state_dict, strict=False)
        
        loaded_adapter_keys = [k for k in new_state_dict.keys() if 'lora' in k or 'gating' in k]
        matched_keys = [k for k in loaded_adapter_keys if k not in incompatible.missing_keys]
        
        if len(matched_keys) > 0:
            print(f"SUCCESS: Loaded {len(matched_keys)} adapter parameters.")
        else:
            print("WARNING: No adapter parameters matched!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--shots', type=int, default=4)
    parser.add_argument('--backbone', type=str, default='ViT-B/16')
    
    parser.add_argument('--adapter', type=str, required=True, choices=['lora', 'ohsinglora', 'singlora'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--r', type=int, default=2)
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--position', type=str, default='all')
    parser.add_argument('--encoder', type=str, default='both')
    parser.add_argument('--params', nargs='+', default=['q', 'k', 'v'])
    
    args = parser.parse_args()
    
    # 1. Load CLIP
    print(f"Loading CLIP {args.backbone}...")
    model, _ = clip.load(args.backbone, device='cuda')
    
    class AdapterArgs:
        adapter = args.adapter
        encoder = args.encoder
        position = args.position
        params = args.params
        r = args.r
        alpha = args.alpha
        ramp_up_steps = 0
        num_heads = args.num_heads
        dropout_rate = 0.0
        backbone = args.backbone

    print(f"Applying structure for {args.adapter}...")
    if args.adapter == 'lora':
        apply_lora(AdapterArgs(), model)
    else:
        apply_adapter(AdapterArgs(), model)
        
    model.cuda()
    
    model.to(model.dtype)
    for name, module in model.named_modules():
        if "LayerNorm" in type(module).__name__:
            module.float()
            
    load_weights_smart(model, args.checkpoint, args.adapter, args)

    print("\n========== ROBUSTNESS EVALUATION ==========")
    results = {}
    dataset_obj = build_dataset(args.dataset, args.root_path, args.shots, None)
    
    for severity in [0, 1, 2, 3]:
        severity_name = ["Clean", "Noise (Light)", "Noise+Blur (Med)", "Hard Core (High)"][severity]
        print(f"\n>> Testing Severity: {severity} ({severity_name})")
        
        transform = get_corrupted_transform(model.visual.input_resolution, severity)
        val_loader = DataLoader(
            DatasetWrapper(dataset_obj.test, input_size=model.visual.input_resolution, transform=transform, is_train=False),
            batch_size=128, shuffle=False, num_workers=4
        )
        
        acc = evaluate(model, val_loader, dataset_obj)
        print(f"Accuracy: {acc:.2f}%")
        results[severity] = acc

    print("\n========== SUMMARY ==========")
    print(f"Model: {args.adapter}")
    print(f"Dataset: {args.dataset}")
    for k, v in results.items():
        print(f"Level {k}: {v:.2f}%")
    
    drop = results[0] - results[3]
    print(f"Performance Drop: {drop:.2f}%")

if __name__ == "__main__":
    main()
    
    
# python3 eval_robustness.py \
#   --dataset eurosat \
#   --root_path /content/DATA \
#   --adapter lora \
#   --checkpoint "/content/drive/MyDrive/RESEARCH/OHSinglora_CLIP/checkpoints_rank2/vitb16/eurosat/4shots/seed1/adapter_weights.pt" \
#   --r 2