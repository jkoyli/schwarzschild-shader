"""
史瓦西黑洞 CPU 参考渲染器 —— 第 1 级（零 GPU 风险）

═══════════════════════════════════════════════════════════════════════════════
为什么先用 CPU 渲染，而不是直接写 shader
═══════════════════════════════════════════════════════════════════════════════
1. **零风险**：纯 numpy 计算，不碰 GPU。着色器写错会挂驱动（TDR），
   Python 写错只会报个异常。
2. **物理先对，画面再对**：CPU 版跑出来的图是"标准答案"。移植到 shader 之后，
   画面一旦有偏差，就能立刻判断是 shader 的 bug 而不是物理理解错了。
   顺序反过来做，这两类错误会混在一起，很难查。
3. **可以慢慢来**：CPU 版渲染一张 640×400 只要几秒，可以反复调参看图，
   而 GPU 版每次改都要担心迭代次数。

═══════════════════════════════════════════════════════════════════════════════
渲染方程
═══════════════════════════════════════════════════════════════════════════════
对每个像素：
  1. 从相机出发构造一条视线（世界坐标 → 单位方向）
  2. 把它化成「轨道平面 + 碰撞参数 b」，用 u(φ) 形式积分测地线
  3. 三种归宿：
       · 落进视界      → 黑
       · 穿过赤道面且落在盘内 → 吸积盘发光（含多普勒/引力红移）
       · 逃逸到无穷远  → 采样星空（这就是"引力透镜扭曲星空"的来源）
  4. 按光线路径顺序做 alpha 合成（盘可以半透明）

【关键：星空扭曲不需要额外做什么】
它**自动**出现 —— 因为逃逸方向是从积分出来的测地线得到的，而不是原始视线方向。
爱因斯坦环、次级环、盘背面绕到上方的那圈光，全都是同一条机制的自然产物。
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schwarzschild import (  # noqa: E402
    RS, SHADOW_RADIUS, ISCO, trace_geodesic, geodesic_u, rk4_step_u,
)

# ── 渲染参数 ────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 640, 400
FOV_Y = 42.0                     # 垂直视场角（度）
SUPERSAMPLE = 2                  # 超采样倍数：**正确性需要**，不只是画质（见 render() 的注释）。
                                 # 代价是像素数 ×N²，所以 2 是 CPU 版的实用上限。

# 【取景：为什么相机要拉远到 40 r_s】
# 第一版用 22 r_s + 盘外缘 14 r_s，结果盘占满整个视场 —— 黑洞阴影和透镜弧
# 反而成了次要元素，画面看着像"一个发光的甜甜圈"而不是黑洞。
# 黑洞的辨识度来自**阴影与透镜弧的比例关系**：阴影要足够小、周围要留出星空，
# 才能看出"星空被弯了"。所以拉远相机、缩小盘的外缘。
CAM_DIST = 40.0                  # 相机到黑洞中心的距离（单位：r_s）
CAM_ELEV_DEG = 9.0               # 相机仰角（0 = 正好在赤道面上）
CAM_AZIM_DEG = 0.0

DISK_INNER = ISCO                # 吸积盘内缘 = 最内稳定圆轨道 3 r_s
DISK_OUTER = 11.0                # 外缘收窄：之前 14 太大，盘会顶到画面边缘
DISK_OPACITY = 0.62              # 降低不透明度，让星空能透出来（也更像薄盘）
# 内缘"视觉温度"（K）。真实吸积盘内缘是上亿 K（峰值在 X 射线），肉眼根本看不见；
# 所有"好看的"黑洞图都把它压到几千~几万 K。这里 20000 K 让内缘落在蓝白、
# 外缘落到橙红，整条盘才有黑体色的**色相梯度** —— 之前 16000 仍然偏白。
DISK_T_ISCO = 20000.0

DPHI = 0.004                     # u(φ) 积分步长
MAX_PHI = 40.0                   # 单条光线的最大转角（防极近 b_c 的光线转不停）


# ═══════════════════════════════════════════════════════════════════════════
#  相机与轨道平面
# ═══════════════════════════════════════════════════════════════════════════
def camera_basis(dist, elev_deg, azim_deg):
    """
    返回 (相机位置, 前, 右, 上)。

    相机永远看向黑洞中心（原点）。用右手系：right × up = -forward（看向原点）。
    """
    e = math.radians(elev_deg)
    a = math.radians(azim_deg)
    pos = np.array([
        dist * math.cos(e) * math.cos(a),
        dist * math.sin(e),            # y 轴向上 = 黑洞自转轴方向
        dist * math.cos(e) * math.sin(a),
    ])
    forward = -pos / np.linalg.norm(pos)          # 指向原点
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-9:              # 正对极轴时的退化情况
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return pos, forward, right, up


def ray_orbital_plane(cam_pos, direction):
    """
    把一条视线化成轨道平面基。

    光子轨道平面由 (r, v) 张成：r 是相机相对黑洞的位置向量，v 是视线方向。
    在这个平面内取极坐标，光线的运动就完全由 u(φ) 描述。

    返回 (e1, e2, b, phi_offset)：
        e1 —— 平面内指向相机的方向（φ=0 的基准）
        e2 —— 平面内垂直于 e1 的方向（决定 φ 增大的旋向）
        b  —— 碰撞参数 = |r × v|
    返回 None 表示这条光线正对中心（b≈0），此时直接判为落入视界。
    """
    r = cam_pos
    r_len = np.linalg.norm(r)
    e1 = r / r_len

    # b = |r × v|：碰撞参数（不依赖 |v|，因为 v 是单位向量）
    cross = np.cross(r, direction)
    b = float(np.linalg.norm(cross))
    if b < 1e-9:
        return None

    # e2：平面内与 e1 垂直、且与光线前进方向同侧
    e2 = np.cross(cross / b, e1)
    e2 /= np.linalg.norm(e2)
    # 确保是"朝黑洞飞"的那一侧：视线方向在 e2 上的投影应当为负（指向 -e1 一侧）
    if np.dot(direction, e2) > 0:
        e2 = -e2
    return e1, e2, b, r_len


# ═══════════════════════════════════════════════════════════════════════════
#  吸积盘：黑体色 + 多普勒红移
# ═══════════════════════════════════════════════════════════════════════════
def blackbody_rgb(temp_k):
    """
    黑体色 → 线性 RGB（0~1）。用的是 Tanner Helland 的经典分段拟合，
    这是业内广泛使用的近似式，误差在视觉范围内可忽略。

    注意：它只给**颜色**，不给亮度 —— 亮度由温度剖面单独算（见 disk_color）。
    """
    t = max(1000.0, min(40000.0, temp_k)) / 100.0

    if t <= 66.0:
        r = 255.0
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)

    if t <= 66.0:
        g = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)

    if t >= 66.0:
        b = 255.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10.0) - 305.0447927307

    return (max(0.0, min(1.0, r / 255.0)),
            max(0.0, min(1.0, g / 255.0)),
            max(0.0, min(1.0, b / 255.0)))


def disk_emission(r, phi_world, cam_pos, incoming_dir):
    """
    吸积盘上半径 r、世界方位角 phi_world 处发出的光，回到相机时是什么颜色和亮度。

    三件事叠加：
      1. **温度剖面** T ∝ r^(-3/4)（Shakura–Sunyaev 薄盘的经典结果）
      2. **多普勒频移**：盘物质以开普勒速度绕转，朝向观察者的一侧变亮变蓝，
         远离的一侧变暗变红 —— 这就是"一侧亮一侧暗"的流光感的来源
      3. **引力红移**：越靠近视界，光逃出来损失的能量越多

    返回线性 RGB（可能 > 1，交由后面的色调映射处理）。
    """
    if r < DISK_INNER or r > DISK_OUTER:
        return None

    # 1) 温度剖面
    t_emit = DISK_T_ISCO * (r / DISK_INNER) ** (-0.75)

    # 2) 开普勒轨道速度（史瓦西，圆轨道）：v = sqrt(M/r)，取 r_s = 2M = 1 → M = 0.5
    #    相对论形式：v = sqrt(M/(r - 2M))·… 这里用几何单位下的常用式
    m = 0.5
    v_orb = math.sqrt(m / max(r, 1.0001))
    # 轨道切向（绕 +y 轴）
    tangential = np.array([-math.sin(phi_world), 0.0, math.cos(phi_world)])
    # 光子离开盘的方向 = incoming_dir 的反向（我们从相机反向追，所以 incoming 是其逆）
    photon_out = -np.asarray(incoming_dir)
    cos_theta = float(np.dot(tangential, photon_out))
    beta = min(0.95, v_orb)
    gamma = 1.0 / math.sqrt(max(1e-6, 1.0 - beta * beta))
    doppler = 1.0 / (gamma * (1.0 - beta * cos_theta))

    # 3) 引力红移
    grav = math.sqrt(max(1e-6, 1.0 - RS / r))

    t_obs = t_emit * doppler * grav

    # 亮度：黑体的总辐射 ∝ T⁴，但那样动态范围太大（内缘比外缘亮上万倍），
    # 视觉上必须压缩。这里连着调了三轮，把结论记下来：
    #   T^2.0  → 内缘直接饱和，整条盘一片白，黑体色的层次全丢
    #   T^1.4  → 好一些，但多普勒不对称仍被压平（亮侧同样饱和）
    #   T^1.0  → 现在这个：亮度与温度成正比，动态范围收得住，
    #            "亮侧 vs 暗侧"和"内缘 vs 外缘"两个梯度都留得住
    # 另外整体乘 0.35 —— 盘是发光体，但它的绝对亮度不该压过黑洞结构本身。
    bright = (t_emit / DISK_T_ISCO) ** 1.0
    bright *= doppler ** 3.0
    bright *= 0.35

    col = blackbody_rgb(t_obs)
    return (col[0] * bright, col[1] * bright, col[2] * bright)


# ═══════════════════════════════════════════════════════════════════════════
#  星空：程序化生成（不依赖任何贴图文件）
# ═══════════════════════════════════════════════════════════════════════════
def _hash3(x, y, z):
    """整数哈希 → [0,1)。用于程序化星场，确定性可复现。"""
    n = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791)
    n = (n ^ (n >> 13)) * 1274126177
    n = n ^ (n >> 16)
    return (n & 0x7FFFFFFF) / float(0x7FFFFFFF)


def _star_dir(face, gx, gy, grid, sub_x, sub_y):
    """
    把「立方体面 + 格坐标」还原成一个**单位方向向量**。

    这是修正星星画法的关键一步：之前直接在面 UV 上量距离，而面 UV 的局部尺度
    在不同面、不同位置上差别极大（面中心 vs 角落能差 2 倍以上），
    于是"等距"的星星被画成了放射状条纹。改成先还原方向、再用方向之间的夹角度量距离，
    尺度就统一了。
    """
    # 格中心 + 亚格偏移 → 面 UV 的 [-1,1]
    u = ((gx + sub_x) / grid) * 2.0 - 1.0
    v = ((gy + sub_y) / grid) * 2.0 - 1.0
    # 反投影：六个面各自的 (u,v) → 方向
    if face == 0:
        d = (1.0, u, v)
    elif face == 1:
        d = (-1.0, u, v)
    elif face == 2:
        d = (u, 1.0, v)
    elif face == 3:
        d = (u, -1.0, v)
    elif face == 4:
        d = (u, v, 1.0)
    else:
        d = (u, v, -1.0)
    n = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    return np.array([d[0] / n, d[1] / n, d[2] / n])


def starfield(direction, pixel_angle=None):
    """
    给定单位方向，返回星空颜色（线性 RGB）。

    做法：把天球按立方体投影分成格子，每格最多一颗星；星的位置/亮度/色温由哈希决定；
    再加一层淡淡的银河带。

    【星的半径必须用**角度**衡量】
    一颗星在屏幕上的大小 ≈ 1 像素，对应角度 = FOV / 分辨率。
    所以判据是「视线方向与星方向的夹角 < ~1 像素对应角」，
    用两个单位向量的叉积模长（= sin θ）来算，天然与画面分辨率无关。
    """
    if pixel_angle is None:
        # 默认按当前渲染设置估算一像素对应的角度
        if HEIGHT > 0:
            pixel_angle = math.radians(FOV_Y) / HEIGHT
        else:
            pixel_angle = 0.002

    d = np.asarray(direction, dtype=float)
    d = d / max(1e-12, np.linalg.norm(d))

    # 立方体投影：取绝对值最大的分量作为主面
    ax = np.abs(d)
    if ax[0] >= ax[1] and ax[0] >= ax[2]:
        uv = (d[1] / ax[0], d[2] / ax[0])
        face = 0 if d[0] > 0 else 1
    elif ax[1] >= ax[2]:
        uv = (d[0] / ax[1], d[2] / ax[1])
        face = 2 if d[1] > 0 else 3
    else:
        uv = (d[0] / ax[2], d[1] / ax[2])
        face = 4 if d[2] > 0 else 5

    # 银河带：沿 y≈0 平面附近的一条淡光带
    #
    # 【这里踩过一个坑：正弦频率选高了会变成"隐约的斑块"】
    # 原来用 sin(dx*9.1)*sin(dz*7.7)*sin((dx+dz)*3.3)，看着像"絮状结构"，
    # 但那几个频率在 42° 视场里对应约 7° 的周期 —— 一提曝光就变成一片
    # 缓慢起伏的斑块，看起来像脏东西，而不是银河。
    # 银河的真实结构尺度是**几十度**量级，所以频率要压到 1~2 才对。
    band = math.exp(-(d[1] * d[1]) / (2 * 0.16 * 0.16))
    texture = 0.62 + 0.38 * math.sin(d[0] * 1.35 + 0.7) * math.sin(d[2] * 1.15 - 1.1)
    base = band * texture * 0.020
    col = np.array([base * 0.66, base * 0.74, base * 1.0])   # 银河偏冷蓝

    # 两层星：亮星稀疏、暗星密集
    #
    # 【密度是这里最容易算错的地方 —— 我连着错了三次，把教训写全】
    # 错法一：网格 220（每格差不到 2 像素）+ 13% 密度 → 每 3×4 像素一颗星。
    #         那不是星空是噪点；被引力透镜一扭就成放射状条纹，
    #         还让我误判成"欠采样"，白加了一轮超采样。
    # 错法二：以为"格子数 = 360/角间隔 = 240"。但天球是立方体六面投影，
    #         每面只覆盖 90°，所以 **每面格子数 = 240/4 = 60** 才对。
    # 错法三：60 格仍然偏密（1.5°/颗，画面上约每 20 像素一颗，还是雪花）。
    # 现在：亮星每面 90 格（1°）、暗星每面 260 格，且亮度用**帕累托分布**
    #       （少数亮星 + 大量暗星）—— 真实星空就是这个统计特征，
    #       等亮度分布会显得像人工噪点。
    for layer, (grid, dens, maxamp, rad_px) in enumerate((
            (70.0, 0.14, 6.0, 0.58),      # 亮星层：每面 70 格 ≈ 1.3°
            (190.0, 0.22, 0.30, 0.50),    # 暗星层：更密但很暗，只提供"底噪星星"
    )):
        gx = int(math.floor((uv[0] + 1.0) * 0.5 * grid))
        gy = int(math.floor((uv[1] + 1.0) * 0.5 * grid))
        if _hash3(gx, gy, face + layer * 517) >= dens:
            continue
        sx = _hash3(gx, gy, face + 101 + layer * 13)
        sy = _hash3(gx, gy, face + 202 + layer * 13)
        sd = _star_dir(face, gx, gy, grid, sx, sy)

        # sin(夹角) —— 这是与分辨率无关的角度度量
        cx = d[1] * sd[2] - d[2] * sd[1]
        cy = d[2] * sd[0] - d[0] * sd[2]
        cz = d[0] * sd[1] - d[1] * sd[0]
        sin_theta = math.sqrt(cx * cx + cy * cy + cz * cz)

        sigma = pixel_angle * rad_px
        core = math.exp(-(sin_theta * sin_theta) / (2.0 * sigma * sigma))
        if core < 3e-3:
            continue
        amp = _hash3(gx, gy, face + 303 + layer * 13)
        # 帕累托式亮度分布：t^3 让大多数星很暗、少数星很亮。
        # 等亮度分布（直接线性用 amp）看起来像人工噪点，因为真实星空的亮星极少。
        amp = amp ** 3.0
        temp_mix = _hash3(gx, gy, face + 404 + layer * 13)   # 0=偏红 1=偏蓝
        star_col = np.array([
            1.0 - 0.45 * temp_mix,
            1.0 - 0.10 * abs(temp_mix - 0.5) * 2.0,
            0.70 + 0.45 * temp_mix,
        ])
        col = col + star_col * core * (0.12 + 0.88 * amp) * maxamp

    return col


# ═══════════════════════════════════════════════════════════════════════════
#  单条光线：积分 + 一次穿过盘就采样
# ═══════════════════════════════════════════════════════════════════════════
def trace_and_shade(cam_pos, direction, e1, e2, b, r_start, pixel_angle=None):
    """
    在轨道平面内积分测地线，并在途中检测与吸积盘的交点。

    这是整个渲染器的核心。做法：
      · 每一步除了推进 u(φ)，还把当前点换算回**世界坐标**，检查它是否穿过赤道面（y=0）
      · 穿过时取交点半径 r，若在盘范围内就采一次样
      · 由于盘可能被多次穿过（绕到背面、下一阶像），按从近到远累积 alpha

    返回线性 RGB。
    """
    u = 0.0
    w = 1.0 / b
    phi = 0.0
    inv_h = 1.0 / RS

    # 上一步的世界坐标（用来判断是否跨过 y=0）
    def world_of(uu, ph):
        rr = 1.0 / max(uu, 1e-12)
        return (math.cos(ph) * e1 + math.sin(ph) * e2) * rr

    prev_pt = world_of(u, phi)
    color = np.zeros(3)
    transmit = 1.0                  # 剩余透射率
    disk_hits = 0

    while phi < MAX_PHI:
        u, w = rk4_step_u(u, w, DPHI)
        phi += DPHI

        if u >= inv_h:
            # 落进视界：已累积的颜色保留，剩余为黑
            return color

        if u <= 0.0:
            # 逃逸：采样星空（这一步就是"引力透镜扭曲星空"）
            # 逃逸方向 = φ 处的切向，在轨道平面内
            vdir = -math.sin(phi) * e1 + math.cos(phi) * e2
            star = starfield(vdir, pixel_angle)
            if disk_hits == 0 and transmit > 0.999:
                return star
            color = color + transmit * star
            return color

        pt = world_of(u, phi)
        # 跨赤道面检测（y 变号）
        if prev_pt[1] * pt[1] < 0.0:
            # 线性插值出交点
            t = prev_pt[1] / (prev_pt[1] - pt[1])
            hit = prev_pt + (pt - prev_pt) * t
            r_hit = math.sqrt(hit[0] ** 2 + hit[1] ** 2 + hit[2] ** 2)
            if DISK_INNER <= r_hit <= DISK_OUTER:
                phi_world = math.atan2(hit[2], hit[0])
                # 光子离开盘的方向（我们反向追，所以取前进方向）
                vdir = -math.sin(phi) * e1 + math.cos(phi) * e2
                emis = disk_emission(r_hit, phi_world, cam_pos, vdir)
                if emis is not None:
                    disk_hits += 1
                    a = DISK_OPACITY
                    color = color + transmit * a * np.array(emis)
                    transmit *= (1.0 - a)
                    if transmit < 0.01:
                        return color
        prev_pt = pt

    # 步数/转角用尽：按捕获处理，但保留已累积的盘光（高阶像常常在这里被截断）
    return color


# ═══════════════════════════════════════════════════════════════════════════
#  主渲染循环
# ═══════════════════════════════════════════════════════════════════════════
def render():
    """
    渲染主循环。

    ═══════════════════════════════════════════════════════════════════════════
    【为什么必须超采样 —— 这不是"画质选项"，是正确性问题】
    ═══════════════════════════════════════════════════════════════════════════
    在强透镜区（b 接近临界值 2.598 的那一圈），光线绕黑洞转的圈数对 b 极其敏感：
    b 变化 0.1%，phi_total 就变化约 1 弧度。这意味着**相邻像素的逃逸方向可以相差很大**。

    每像素只采一次样时，星场的亮暗就被随机"抽"到不同的星上，画出来是一圈放射状条纹
    （第一次渲染就是这个结果，看着像噪声，其实是欠采样）。真实的黑洞照片里
    阴影边缘那圈"细密的同心弧"就是这个效应的**正确**版本 —— 只有把采样做够才看得见。

    所以这里按 SUPERSAMPLE 倍渲染再平均。代价是像素数 ×N²，但换来的是正确的画面。
    ═══════════════════════════════════════════════════════════════════════════
    """
    cam_pos, forward, right, up = camera_basis(CAM_DIST, CAM_ELEV_DEG, CAM_AZIM_DEG)

    ss = max(1, int(SUPERSAMPLE))
    rw, rh = WIDTH * ss, HEIGHT * ss

    aspect = WIDTH / float(HEIGHT)
    half_h = math.tan(math.radians(FOV_Y) * 0.5)
    half_w = half_h * aspect

    acc = np.zeros((HEIGHT, WIDTH, 3), dtype=float)
    pixel_angle = math.radians(FOV_Y) / rh      # 子像素对应的角度（星点半径要用它）

    for j in range(HEIGHT):
        for i in range(WIDTH):
            rgb = np.zeros(3)
            # 子像素均匀抖动网格（stratified）：比单纯 4 角平均更抗走样
            for sj in range(ss):
                for si in range(ss):
                    fx = (i + (si + 0.5) / ss) / WIDTH
                    fy = (j + (sj + 0.5) / ss) / HEIGHT
                    sy = (1.0 - 2.0 * fy) * half_h
                    sx = (2.0 * fx - 1.0) * half_w
                    direction = forward + right * sx + up * sy
                    direction = direction / np.linalg.norm(direction)

                    plane = ray_orbital_plane(cam_pos, direction)
                    if plane is None:
                        continue
                    e1, e2, b, r_start = plane
                    rgb += trace_and_shade(cam_pos, direction, e1, e2, b, r_start,
                                           pixel_angle)
            acc[j, i] = rgb / (ss * ss)
        if (j + 1) % 10 == 0 or j == HEIGHT - 1:
            print('  行 %3d/%d  (%.1f%%)' % (j + 1, HEIGHT, 100.0 * (j + 1) / HEIGHT),
                  flush=True)

    return acc, (cam_pos, forward, right, up)


# ═══════════════════════════════════════════════════════════════════════════
#  色调映射 + PNG 输出（不依赖 Pillow，自己写 zlib PNG）
# ═══════════════════════════════════════════════════════════════════════════
def tonemap(img, exposure=1.0):
    """
    线性 HDR → 8 位 sRGB。

    顺序很重要：**先曝光 → 再色调映射 → 最后 gamma**。
    反过来（先 gamma 再相加/映射）会让亮部糊成一坨白，这是后处理最常见的错。
    """
    x = np.maximum(0.0, img) * exposure
    # Reinhard 色调映射：x/(1+x)，把无界亮度压到 [0,1)
    x = x / (1.0 + x)
    # 线性 → sRGB
    x = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.maximum(x, 1e-12), 1 / 2.4) - 0.055)
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def write_png(path, arr):
    """写 8 位 RGB PNG。自己实现是为了不引入 Pillow 依赖。"""
    import struct
    import zlib

    h, w, _ = arr.shape
    raw = bytearray()
    for y in range(h):
        raw.append(0)                      # 每行的 filter type = 0
        raw.extend(arr[y].tobytes())

    def chunk(tag, data):
        c = struct.pack('>I', len(data)) + tag + data
        c += struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF)
        return c

    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(bytes(raw), 9))
    png += chunk(b'IEND', b'')

    with open(path, 'wb') as f:
        f.write(png)


def main():
    print('史瓦西黑洞 CPU 参考渲染')
    print('  分辨率 %dx%d  相机距离 %.1f r_s  仰角 %.1f°'
          % (WIDTH, HEIGHT, CAM_DIST, CAM_ELEV_DEG))
    print('  吸积盘 %.1f ~ %.1f r_s' % (DISK_INNER, DISK_OUTER))
    print('')

    img, _ = render()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
    os.makedirs(out_dir, exist_ok=True)

    # 存两份：曝光 1.0 和更高曝光（盘的外缘很暗，需要提亮才看得清）
    for tag, exp in (('exposure1', 1.0), ('exposure3', 3.0)):
        p = os.path.join(out_dir, 'blackhole_%s.png' % tag)
        write_png(p, tonemap(img, exp))
        print('  已写出 %s' % p)

    # 顺手存线性浮点数据，方便后面调色调映射而不用重渲染
    np.save(os.path.join(out_dir, 'blackhole_linear.npy'), img)
    print('  已写出线性数据 blackhole_linear.npy（可反复调色调映射，不必重渲）')


if __name__ == '__main__':
    main()
