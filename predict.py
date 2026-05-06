import os
import sys
import argparse
from glob import glob

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# === 預設路徑 ===
BASE = '/content/drive/MyDrive/Colab Notebooks/NTU_notebooks/master_reaserch/CloudAdapter'
DEFAULT_MODEL_DIR  = os.path.join(BASE, 'model')
DEFAULT_CONFIG     = os.path.join(BASE, 'model',      'multi_classes_512x512.py')
DEFAULT_CKPT       = os.path.join(BASE, 'checkpoint', 'l1c_full_weight.pth')
DEFAULT_INPUT_DIR  = os.path.join(BASE, 'data')
DEFAULT_OUTPUT_DIR = os.path.join(BASE, 'outputs')

# === Palette + class names ===
PALETTE_L1C    = [79, 253, 199, 77, 2, 115, 251, 255, 41, 221, 53, 223]
PALETTE_L8     = [79, 253, 199, 221, 53, 223, 251, 255, 41, 77, 2, 115]
PALETTE_BINARY = [79, 253, 199, 77, 2, 115]

CLASS_NAMES_L1C    = ['Clear Sky', 'Thick Cloud', 'Thin Cloud', 'Cloud Shadow']
CLASS_NAMES_L8     = ['Clear Sky', 'Cloud Shadow', 'Thin Cloud', 'Thick Cloud']
CLASS_NAMES_BINARY = ['Clear', 'Cloud']

PROFILES = {
    'l1c':    dict(palette=PALETTE_L1C,    class_names=CLASS_NAMES_L1C,    img_size=512),
    'l8':     dict(palette=PALETTE_L8,     class_names=CLASS_NAMES_L8,     img_size=512),
    'binary': dict(palette=PALETTE_BINARY, class_names=CLASS_NAMES_BINARY, img_size=256),
}


def load_model(config_path, ckpt_path, model_dir=DEFAULT_MODEL_DIR, device=None):
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from mmseg.apis import init_model
    from cloud_adapter.cloud_adapter_dinov2 import CloudAdapterDinoVisionTransformer  # noqa: F401

    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'[init] config = {config_path}')
    print(f'[init] ckpt   = {ckpt_path}')
    print(f'[init] device = {device}')
    model = init_model(config_path, ckpt_path, device=device)
    model.eval()
    return model, device


def make_visualization(input_image, pred_indices, palette, class_names,
                        save_path, clear_class=0, alpha=0.5):
    """input / 預測 / overlay 三聯圖。pred_indices: HxW class index, 解析度需與 input 相同。"""
    rgb_palette = np.array(palette, dtype=np.uint8).reshape(-1, 3)
    n_cls = len(class_names)

    img_arr    = np.array(input_image.convert('RGB'))
    color_mask = rgb_palette[pred_indices]                 # HxWx3

    overlay = img_arr.copy()
    blend = pred_indices != clear_class
    overlay[blend] = ((1 - alpha) * img_arr[blend] +
                      alpha * color_mask[blend]).astype(np.uint8)

    pred_legend    = [mpatches.Patch(color=rgb_palette[i] / 255,
                                     label=f'{i} = {class_names[i]}') for i in range(n_cls)]
    overlay_legend = [mpatches.Patch(color=rgb_palette[i] / 255,
                                     label=f'Predicted {class_names[i]}')
                      for i in range(n_cls) if i != clear_class]

    fig, axes = plt.subplots(1, 3, figsize=(18, 9))
    axes[0].imshow(input_image); axes[0].set_title('Input');      axes[0].axis('off')
    axes[1].imshow(color_mask);  axes[1].set_title('Prediction'); axes[1].axis('off')
    axes[1].legend(handles=pred_legend, loc='lower right', fontsize=10)
    axes[2].imshow(overlay);     axes[2].set_title('Overlay');    axes[2].axis('off')
    axes[2].legend(handles=overlay_legend, loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def predict_one(model, image_path, mask_path, device,
                img_size=512, palette=PALETTE_L1C, class_names=None,
                visualize=True, viz_path=None):
    img = Image.open(image_path).convert('RGB')
    ori_size = img.size

    x = img.resize((img_size, img_size), resample=Image.Resampling.BILINEAR)
    x = np.array(x).astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    x = torch.from_numpy(x).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)

    outs = model.predict(x)
    pred = outs[0].pred_sem_seg.data.cpu().numpy().astype(np.uint8)[0]

    mask = Image.fromarray(pred).convert('P')
    mask.putpalette(palette)
    mask_full = mask.resize(ori_size, resample=Image.Resampling.NEAREST)
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    mask_full.convert('RGB').save(mask_path)

    if visualize and viz_path:
        pred_full = np.array(Image.fromarray(pred).resize(ori_size,
                                                          resample=Image.Resampling.NEAREST))
        os.makedirs(os.path.dirname(viz_path), exist_ok=True)
        names = class_names or [f'class {i}' for i in range(len(palette) // 3)]
        make_visualization(img, pred_full, palette, names, viz_path)

    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return mask_full


def predict(
    config_path=DEFAULT_CONFIG,
    ckpt_path=DEFAULT_CKPT,
    input_path=DEFAULT_INPUT_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    mask_dir=None,        # 預設 = output_dir/mask
    viz_dir=None,         # 預設 = output_dir/viz
    model_dir=DEFAULT_MODEL_DIR,
    img_size=512,
    palette=PALETTE_L1C,
    class_names=CLASS_NAMES_L1C,
    visualize=True,
    device=None,
    model=None,
):
    mask_dir = mask_dir or os.path.join(output_dir, 'mask')
    viz_dir  = viz_dir  or os.path.join(output_dir, 'viz')
    os.makedirs(mask_dir, exist_ok=True)
    if visualize:
        os.makedirs(viz_dir, exist_ok=True)

    if model is None:
        model, device = load_model(config_path, ckpt_path, model_dir=model_dir, device=device)
    elif device is None:
        device = next(model.parameters()).device.type

    if os.path.isdir(input_path):
        exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        paths = sorted(p for p in glob(os.path.join(input_path, '*')) if p.lower().endswith(exts))
    else:
        paths = [input_path]

    if not paths:
        print(f'[warn] no images found in {input_path}')
        return []

    results = []
    for i, p in enumerate(paths, 1):
        stem = os.path.splitext(os.path.basename(p))[0]
        mask_path = os.path.join(mask_dir, f'{stem}.png')
        viz_path  = os.path.join(viz_dir,  f'{stem}.png') if visualize else None
        predict_one(model, p, mask_path, device,
                    img_size=img_size, palette=palette, class_names=class_names,
                    visualize=visualize, viz_path=viz_path)
        print(f'[{i}/{len(paths)}] {p}\n         mask -> {mask_path}\n         viz  -> {viz_path}')
        results.append((p, mask_path, viz_path))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',    default=DEFAULT_CONFIG)
    parser.add_argument('--ckpt',      default=DEFAULT_CKPT)
    parser.add_argument('--input',     default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output',    default=DEFAULT_OUTPUT_DIR,
                        help='會自動建立 output/mask 與 output/viz')
    parser.add_argument('--mask-dir',  default=None, help='覆寫預設 output/mask')
    parser.add_argument('--viz-dir',   default=None, help='覆寫預設 output/viz')
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR)
    parser.add_argument('--profile',   choices=list(PROFILES), default='l1c')
    parser.add_argument('--no-viz',    action='store_true')
    args = parser.parse_args()

    prof = PROFILES[args.profile]
    predict(
        config_path=args.config,
        ckpt_path=args.ckpt,
        input_path=args.input,
        output_dir=args.output,
        mask_dir=args.mask_dir,
        viz_dir=args.viz_dir,
        model_dir=args.model_dir,
        img_size=prof['img_size'],
        palette=prof['palette'],
        class_names=prof['class_names'],
        visualize=not args.no_viz,
    )


if __name__ == '__main__':
    main()