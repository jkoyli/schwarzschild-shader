"""
后处理流水线驱动 —— 用已渲染的线性数据反复调参，不必重渲染黑洞。

用法：
    python compose.py                  # 默认参数
    python compose.py --no-ripple      # 只看泛光
    python compose.py --radius 300     # 指定扩散环半径

【为什么单独一步】
黑洞渲染一次要 90 秒，后处理只要 1 秒。把两者拆开，调颜色/泛光/扩散环时
反复跑的是 1 秒的那步 —— 这就是当初把线性数据存成 .npy 的意义。
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import postfx  # noqa: E402
from render import write_png  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default=None, help='线性 .npy 输入')
    ap.add_argument('--out', dest='out', default=None, help='输出 PNG')
    ap.add_argument('--exposure', type=float, default=None,
                    help='曝光；不填则自适应')
    ap.add_argument('--bloom', type=float, default=0.85, help='泛光强度')
    ap.add_argument('--threshold', type=float, default=0.75, help='亮部软阈值')
    ap.add_argument('--mode', default='aces', choices=['reinhard', 'filmic', 'aces'])
    ap.add_argument('--ripple', dest='ripple', type=float, default=None,
                    help='扩散环半径（像素）；不填则不加')
    ap.add_argument('--ripple-width', type=float, default=15.0)
    ap.add_argument('--ripple-push', type=float, default=8.0)
    ap.add_argument('--ripple-glow', type=float, default=0.30)
    ap.add_argument('--no-bloom', dest='bloom_on', action='store_false', default=True)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    inp = args.inp or os.path.join(here, 'out', 'blackhole_linear.npy')
    out = args.out or os.path.join(here, 'out', 'post.png')

    img = np.load(inp)
    print('读入 %s  %s' % (os.path.basename(inp), img.shape))

    exp = args.exposure
    if exp is None:
        exp = postfx.auto_exposure(img)
        print('自适应曝光 = %.3f' % exp)

    x = img

    if args.bloom_on:
        x = postfx.bloom(x, threshold=args.threshold, strength=args.bloom)
        print('泛光: threshold=%.2f strength=%.2f' % (args.threshold, args.bloom))

    if args.ripple is not None:
        x = postfx.shockwave_ripple(
            x, radius=args.ripple, width=args.ripple_width,
            push=args.ripple_push, glow=args.ripple_glow)
        print('扩散环: 半径=%.0fpx 厚度=%.0fpx 位移=%.0fpx'
              % (args.ripple, args.ripple_width, args.ripple_push))

    rgb = postfx.tonemap(x, exposure=exp, mode=args.mode)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_png(out, rgb)
    print('已写出 %s' % out)


if __name__ == '__main__':
    main()
