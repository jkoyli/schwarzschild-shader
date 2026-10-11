"""
后处理层 —— 曝光 / 泛光（bloom）/ 元素视野式扩散环

═══════════════════════════════════════════════════════════════════════════════
为什么后处理必须独立成一层
═══════════════════════════════════════════════════════════════════════════════
渲染出来的线性 HDR 数据是"物理量"，但它不是"能看的图"。中间差三步：

    线性 HDR → [曝光] → [泛光] → [色调映射] → [gamma] → 屏幕

顺序错一步就会出典型毛病：
  · 在 gamma 空间相加   → 两个 0.5 就到顶，第三个再也加不上去，亮部糊成一坨白
  · 先色调映射再泛光    → 泛光变成"贴上去的一层白纱"，硬边不透光
  · 泛光用 mix 而非加性 → 是"替换"能量不是"叠加"，整屏对比度暴跌、像起雾

═══════════════════════════════════════════════════════════════════════════════
关于"元素视野式扩散环"
═══════════════════════════════════════════════════════════════════════════════
这个效果和黑洞是**两套独立技术**：黑洞是逐像素追测地线（几何问题），
扩散环是屏幕空间的**径向位移 + 环形高光**（图像滤波问题）。所以它放在后处理层。

物理模型很简单：一个冲击波从中心向外扩，波前是一个半径 r_w(t) 的圆环。
波前扫过某个像素时，该像素的采样位置被**沿半径方向往外推**一点；
推得越多，看起来扭曲越强。关键细节：

  1. **主环 + 拖尾**：只用高斯会显得像"一圈面包圈"。真实的冲击波后面跟着
     一圈内凹的低压区，所以用「高斯主环 − 内侧副环」的双瓣剖面。
  2. **位移必须是径向的**：沿半径往外推 → 画面"被推开"；往内推 → "被吸进去"。
     元素视野那种是往外推。
  3. **色差**：R/G/B 用略微不同的位移量，边缘就会出现彩虹描边 —— 这是"能量感"的来源，
     成本几乎为零但效果显著。

在 CPU 上做这个只是为了确认参数好看；真正跑起来要在 GPU 上（屏幕空间操作，GPU 快几千倍）。
"""
import math

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  1. 曝光
# ═══════════════════════════════════════════════════════════════════════════
def auto_exposure(img, target_lum=0.22, min_exp=0.05, max_exp=12.0, percentile=99.0):
    """
    自适应曝光：把一个**高分位数**亮度推到目标值附近。

    【为什么不能用均值 —— 这是黑洞画面特有的坑】
    黑洞画面是"大面积纯黑 + 小面积极亮"（黑洞本身不发光，盘只占一小块）。
    用均值做基准时，均值被大片黑色拉得极低，于是算出一个很大的曝光倍数：

        实测：均值 0.0424 → 曝光 5.43 倍 → 盘全部饱和 → 整条盘变成一片白，
        多普勒的"一侧亮一侧暗"和黑体色的色相梯度**全被冲掉了**。

    这跟物理无关，纯粹是自动曝光的基准选错了。改用 99 分位数后，
    曝光由"画面里最亮的那一小撮"决定，盘的内部层次就能保留下来。
    """
    lum = img.sum(axis=2)
    if lum.size == 0:
        return 1.0
    ref = float(np.percentile(lum, percentile))
    if ref <= 1e-9:
        ref = float(lum.mean())
    if ref <= 1e-9:
        return 1.0
    exp = target_lum * 3.0 / ref          # ×3：分位数推到 target_lum 后，
                                          # 还要给最亮的 1% 留出过曝余量
    return float(max(min_exp, min(max_exp, exp)))


# ═══════════════════════════════════════════════════════════════════════════
#  2. 泛光
# ═══════════════════════════════════════════════════════════════════════════
def _box_blur(a, radius):
    """
    可分离盒式模糊（两次一维滑动平均）。用累积和实现，复杂度与半径无关。

    为什么用盒式而不是高斯做初版：可分离高斯要 2×(2k+1) 次乘法，
    盒式只要 2 次前缀和。做多级模糊、每级只有 5 个 tap 时，盒式的误差可以忽略。
    """
    if radius < 1:
        return a

    def blur1d(x, r):
        n = x.shape[0]
        pad = np.pad(x, ((r, r), (0, 0)), mode='edge')
        csum = np.cumsum(pad, axis=0)
        csum = np.vstack([np.zeros((1, x.shape[1])), csum])
        return (csum[2 * r + 1:2 * r + 1 + n] - csum[:n]) / (2 * r + 1)

    if a.ndim == 2:
        a = a[:, :, None]
    tmp = blur1d(a.reshape(a.shape[0], -1), radius).reshape(a.shape)
    tmp = np.transpose(tmp, (1, 0, 2))
    tmp = blur1d(tmp.reshape(tmp.shape[0], -1), radius).reshape(tmp.shape)
    return np.transpose(tmp, (1, 0, 2)).reshape(a.shape)


def bloom(img, threshold=0.75, knee=0.35, size=26, strength=0.85, iterations=3):
    """
    泛光：把亮部提取出来、模糊、再加性叠回去。

    threshold/knee 是软阈值（soft knee）：亮度在 [threshold-knee, threshold+knee] 之间
    平滑过渡，避免出现"亮度刚好跨过阈值 → 边缘突然出现一圈硬边"的难看台阶。

    模糊用逐级放大半径的方式近似 mip 链：radius, 2×radius, 4×radius 各模糊一次再叠加。
    这样低成本就能得到"大范围柔光 + 局部亮核"的层次，比单纯一次大半径模糊自然得多。
    """
    lum = img.sum(axis=2) / 3.0

    # 软阈值
    lo = threshold - knee
    hi = threshold + knee
    w = np.clip((lum - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)              # smoothstep，让过渡更柔
    bright = img * w[:, :, None]

    acc = np.zeros_like(img)
    total_w = 0.0
    r = max(1, size // 8)
    weight = 1.0
    for _ in range(iterations):
        acc += _box_blur(bright, r) * weight
        total_w += weight
        r *= 2
        weight *= 0.65                        # 大半径的贡献递减，避免整体发灰
    acc /= max(1e-9, total_w)

    return img + acc * strength


# ═══════════════════════════════════════════════════════════════════════════
#  3. 元素视野式扩散环
# ═══════════════════════════════════════════════════════════════════════════
def _sample_bilinear(img, xs, ys):
    """
    双线性采样。xs/ys 是浮点像素坐标数组，形状 (H,W)。
    越界处返回 0（黑），这样扩散环推到画面外时不会出现边缘拉伸。
    """
    h, w = img.shape[:2]
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = xs - x0
    fy = ys - y0
    out = np.zeros(xs.shape + (img.shape[2],), dtype=img.dtype)
    for dy in (0, 1):
        for dx in (0, 1):
            xi = np.clip(x0 + dx, 0, w - 1)
            yi = np.clip(y0 + dy, 0, h - 1)
            wgt = (fx if dx else 1.0 - fx) * (fy if dy else 1.0 - fy)
            inside = ((x0 + dx >= 0) & (x0 + dx < w) & (y0 + dy >= 0) & (y0 + dy < h))
            wgt = wgt * inside
            out += img[yi, xi] * wgt[:, :, None]
    return out


def shockwave_ripple(img, radius, width=14.0, push=7.0, trail=0.55,
                     chroma=0.30, glow=0.35, center=None, aspect=1.0):
    """
    元素视野式向外扩散环。

    参数
    ----
    radius   : 当前波前半径（像素）。动画时让它随时间从 0 增长到覆盖全屏。
    width    : 波前厚度（像素）。越薄越"锐利"，越厚越"柔和"。
    push     : 最大径向位移（像素）。往**外**推为正 —— 画面像被推开。
    trail    : 内侧副瓣强度（0~1）。做成主环−副瓣的双瓣剖面，看起来才像冲击波而不是面包圈。
    chroma   : 色差强度。R/G/B 用略微不同的位移，边缘出现彩虹描边。
    glow     : 波前自身的高光强度（让环"发光"而不只是扭曲）。
    center   : (cx, cy)，默认画面中心。
    aspect   : 为将来适配非方形像素预留。

    返回处理后的图像（线性空间，应该**在色调映射之前**调用）。
    """
    h, w = img.shape[:2]
    if center is None:
        cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    else:
        cx, cy = center

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx = (xx - cx)
    dy = (yy - cy) * aspect
    r = np.sqrt(dx * dx + dy * dy)
    r_safe = np.maximum(r, 1e-6)

    # 双瓣剖面：外侧高斯主环 − 内侧副环（模拟波后低压区）
    g_main = np.exp(-((r - radius) ** 2) / (2.0 * (width * 0.5) ** 2))
    g_trail = np.exp(-((r - (radius - width * 1.15)) ** 2) / (2.0 * (width * 0.9) ** 2))
    profile = g_main - trail * g_trail

    # 位移沿半径向外
    push_px = profile * push

    # 单位方向
    ux = dx / r_safe
    uy = dy / r_safe

    # 色差：三个通道用略微不同的位移量（蓝移最多、红移最少）
    out = np.zeros_like(img)
    shifts = (chroma * 0.0, chroma * 0.5, chroma * 1.0)   # R, G, B 的额外位移比例
    for ch, extra in enumerate(shifts):
        s = push_px * (1.0 + extra)
        xs = xx - ux * s
        ys = yy - uy * s
        sampled = _sample_bilinear(img[:, :, ch:ch + 1], xs, ys)
        out[:, :, ch] = sampled[:, :, 0]

    # 波前高光：环本身发光，用冷暖分层的颜色
    if glow > 0.0:
        ring = np.clip(g_main, 0.0, 1.0)
        rim = np.clip(g_main - 0.5 * g_trail, 0.0, 1.0)
        # 主环偏青蓝，内侧副瓣偏暖 —— 制造"能量扫过"的色相变化
        out[:, :, 0] += ring * glow * 0.55 + rim * glow * 0.85
        out[:, :, 1] += ring * glow * 0.85 + rim * glow * 0.55
        out[:, :, 2] += ring * glow * 1.00 + rim * glow * 0.25

    return out


# ═══════════════════════════════════════════════════════════════════════════
#  4. 色调映射 + gamma
# ═══════════════════════════════════════════════════════════════════════════
def tonemap(img, exposure=1.0, mode='reinhard', white=3.0):
    """
    线性 HDR → 8 位 sRGB。**这一步必须是最后一步。**

    mode:
      'reinhard' —— x/(1+x)，简单、不会过曝，但亮部偏灰
      'filmic'   —— 带白点的 Reinhard 变体，高光收得更干净
      'aces'     —— 近似 ACES 曲线，对比度最好，电影感强（推荐）
    """
    x = np.maximum(0.0, img) * exposure

    if mode == 'reinhard':
        y = x / (1.0 + x)
    elif mode == 'filmic':
        wx = white
        y = (x * (1.0 + x / (wx * wx))) / (1.0 + x)
    else:  # aces 近似（Narkowicz 拟合，业内常用）
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        y = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)

    # 线性 → sRGB
    y = np.where(y <= 0.0031308,
                 y * 12.92,
                 1.055 * np.power(np.maximum(y, 1e-12), 1.0 / 2.4) - 0.055)
    return np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)


def to_uint8(img, exposure=1.0, mode='aces'):
    return tonemap(img, exposure=exposure, mode=mode)
