"""
GLSL ↔ Python 一致性审计

═══════════════════════════════════════════════════════════════════════════════
为什么需要这个
═══════════════════════════════════════════════════════════════════════════════
GPU 版（web/index.html 里的 GLSL）和 CPU 版（cpu/*.py）是**同一套数学的两份实现**。
CPU 版的数学已经过了 19 项数值自检（弱场偏折收敛阶、临界碰撞参数、光子球对数发散），
所以它是"标准答案"。

但 GPU 版**从没跑过** —— 这台机器没有 GLSL 编译器，我也打不开浏览器。
而 GLSL 里的系数如果手滑写错一位（比如 1.5 写成 1.0），**编译器不会报任何错**，
画面只是"稍微不对"：黑洞阴影大小差一点、盘的位置偏一点，肉眼基本看不出来。

所以这里做两件事：
  1. **常量一致性**：把 GLSL 源码里的字面量抠出来，与 Python 的对应值逐个比对
  2. **数值一致性**：在 Python 里**照抄 GLSL 的代码路径**重算一遍关键量
     （rk4 一步、黑体色、相机基），与 CPU 版逐位对齐

第 2 条是关键：它验证的不是"我写的 Python 对不对"，而是
"**GLSL 里那段代码，照字面执行，得到的是不是和已验证的 CPU 版一样的结果**"。

用法: python parity_check.py
"""
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HTML = os.path.join(ROOT, 'web', 'index.html')
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from schwarzschild import RS, ISCO, geodesic_u, rk4_step_u  # noqa: E402
import render as R  # noqa: E402

passed = 0
failed = 0
failures = []


def ok(name, cond, detail=''):
    global passed, failed
    if cond:
        passed += 1
        print('  [OK]   %s%s' % (name, ('  ' + detail) if detail else ''))
    else:
        failed += 1
        failures.append(name)
        print('  [FAIL] %s%s' % (name, ('  ' + detail) if detail else ''))


def strip_comments(s):
    return re.sub(r'//[^\n]*', ' ', re.sub(r'/\*[\s\S]*?\*/', ' ', s))


html = open(HTML, encoding='utf-8').read()
shaders = {}
for mm in re.finditer(r'const\s+([A-Z_][A-Z0-9_]*)\s*=\s*`([\s\S]*?)`;', html):
    if '#version' in mm.group(2):
        shaders[mm.group(1)] = mm.group(2)

CODE = strip_comments('\n'.join(shaders.values()))

print('')
print('=' * 66)
print('读入 web/index.html  (%d 字节, %d 个着色器: %s)'
      % (len(html), len(shaders), ', '.join(shaders.keys())))
print('=' * 66)

# ── 1. 测地线方程系数 ─────────────────────────────────────────────────────
print('')
print('=== 1. 测地线方程系数（这一行就是全部物理） ===')
geod = re.search(r'1\.5\s*\*\s*u\s*\*\s*u\s*-\s*u', CODE)
ok('GLSL 里系数是 1.5  (u\'\' = 1.5u² - u)', geod is not None,
   '找到: ' + (geod.group(0) if geod else '没找到'))

wrong = [p for p in (r'\b1\.0\s*\*\s*u\s*\*\s*u\s*-\s*u',
                     r'\b2\.0\s*\*\s*u\s*\*\s*u\s*-\s*u',
                     r'\b0\.5\s*\*\s*u\s*\*\s*u\s*-\s*u') if re.search(p, CODE)]
ok('没有其他版本的系数（1.0 / 2.0 / 0.5）', not wrong, str(wrong))
ok('Python 侧同一系数', abs(geodesic_u(0.3, 0.0) - (1.5 * 0.09 - 0.3)) < 1e-15,
   'u=0.3 → %.12f' % geodesic_u(0.3, 0.0))

# ── 2. RK4 一步：GLSL vs Python ───────────────────────────────────────────
print('')
print('=== 2. RK4 一步：照抄 GLSL 的实现，与 Python 版逐位比对 ===')


def rk4_glsl(u, w, dphi):
    """逐行照抄 web/index.html 里的 rk4()。"""
    k1u = w
    k1w = 1.5 * u * u - u
    u2 = u + 0.5 * dphi * k1u
    w2 = w + 0.5 * dphi * k1w
    k2u = w2
    k2w = 1.5 * u2 * u2 - u2
    u3 = u + 0.5 * dphi * k2u
    w3 = w + 0.5 * dphi * k2w
    k3u = w3
    k3w = 1.5 * u3 * u3 - u3
    u4 = u + dphi * k3u
    w4 = w + dphi * k3w
    k4u = w4
    k4w = 1.5 * u4 * u4 - u4
    return (u + (dphi / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u),
            w + (dphi / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w))


cases = [(0.0, 0.4, 0.004), (0.35, 0.1, 0.004), (0.62, -0.08, 0.02),
         (0.001, 0.999, 0.05), (0.9, -0.5, 0.01)]
max_d = 0.0
for u0, w0, d in cases:
    up, wp = rk4_step_u(u0, w0, d)
    ug, wg = rk4_glsl(u0, w0, d)
    max_d = max(max_d, abs(up - ug), abs(wp - wg))
ok('五种初值下两种 RK4 完全一致（差 < 1e-12）', max_d < 1e-12, '最大差 %.3e' % max_d)

# ── 3. 特征半径常量 ───────────────────────────────────────────────────────
print('')
print('=== 3. 特征半径常量 ===')
m = re.search(r'const\s+float\s+RS\s*=\s*([0-9.]+)\s*;', CODE)
ok('GLSL 里 RS = 1.0', m is not None and abs(float(m.group(1)) - RS) < 1e-12,
   ('找到 ' + m.group(1)) if m else '没找到')
ok('视界判据写成 u >= 1.0/RS', re.search(r'u\s*>=\s*1\.0\s*/\s*RS', CODE) is not None)
ok('逃逸判据是 u <= 0.0', re.search(r'u\s*<=\s*0\.0', CODE) is not None)

host_inner = re.search(r'uDiskInner\s*,\s*([0-9.]+)\s*\)', html)
ok('宿主把盘内缘上传为 3.0（= ISCO = 3 r_s）',
   host_inner is not None and abs(float(host_inner.group(1)) - ISCO) < 1e-12,
   ('上传值 ' + host_inner.group(1)) if host_inner else '没找到')

m2 = re.search(r'phi\s*>\s*([0-9.]+)', CODE)
ok('有转角上限（防光线绕不停）', m2 is not None,
   ('phi > ' + m2.group(1)) if m2 else '没找到')

# ── 4. 吸积盘物理 ─────────────────────────────────────────────────────────
print('')
print('=== 4. 吸积盘：温度剖面 / 开普勒速度 / 多普勒 / 引力红移 ===')
m = re.search(r'pow\s*\(\s*rh\s*/\s*uDiskInner\s*,\s*(-?[0-9.]+)\s*\)', CODE)
ok('温度剖面指数 -0.75  (T ∝ r^(-3/4))',
   m is not None and abs(float(m.group(1)) + 0.75) < 1e-12,
   ('指数 ' + m.group(1)) if m else '没找到')

m = re.search(r'sqrt\s*\(\s*([0-9.]+)\s*/\s*max\s*\(\s*rh\s*,\s*([0-9.]+)\s*\)\s*\)', CODE)
ok('开普勒速度 sqrt(0.5/r)  (M = 0.5 对应 r_s = 1)',
   m is not None and abs(float(m.group(1)) - 0.5) < 1e-12,
   ('分子 ' + m.group(1) + '，r 下限 ' + m.group(2)) if m else '没找到')

ok('多普勒 1/(gamma*(1-beta*cos))',
   re.search(r'1\.0\s*/\s*\(\s*gamma\s*\*\s*\(\s*1\.0\s*-\s*beta\s*\*\s*cosT\s*\)\s*\)', CODE) is not None)
ok('gamma = 1/sqrt(1-beta²)',
   re.search(r'1\.0\s*/\s*sqrt\s*\(\s*max\s*\(\s*1e-6\s*,\s*1\.0\s*-\s*beta\s*\*\s*beta\s*\)\s*\)', CODE) is not None)
ok('beta 上限 0.95（避免除零）', re.search(r'min\s*\(\s*0\.95\s*,\s*vOrb\s*\)', CODE) is not None)
ok('引力红移 sqrt(1 - RS/r)',
   re.search(r'sqrt\s*\(\s*max\s*\(\s*1e-6\s*,\s*1\.0\s*-\s*RS\s*/\s*rh\s*\)\s*\)', CODE) is not None)

m = re.search(r'pow\s*\(\s*tEmit\s*/\s*uDiskT\s*,\s*([0-9.]+)\s*\)', CODE)
ok('亮度幂次 1.0（与 CPU 版一致）',
   m is not None and abs(float(m.group(1)) - 1.0) < 1e-12,
   ('幂次 ' + m.group(1)) if m else '没找到')

m = re.search(r'pow\s*\(\s*doppler\s*,\s*([0-9.]+)\s*\)', CODE)
ok('相对论增亮幂次 3.0', m is not None and abs(float(m.group(1)) - 3.0) < 1e-12,
   ('幂次 ' + m.group(1)) if m else '没找到')

m = re.search(r'pow\s*\(\s*doppler\s*,\s*3\.0\s*\)\s*\*\s*([0-9.]+)', CODE)
ok('整体亮度系数 0.35（与 CPU 版一致）',
   m is not None and abs(float(m.group(1)) - 0.35) < 1e-12,
   ('系数 ' + m.group(1)) if m else '没找到')

# ── 5. 黑体色 ─────────────────────────────────────────────────────────────
print('')
print('=== 5. 黑体色公式：GLSL 版 vs Python 版 ===')


def blackbody_glsl(t):
    t = max(1000.0, min(40000.0, t)) / 100.0
    r = 255.0 if t <= 66.0 else 329.698727446 * ((t - 60.0) ** -0.1332047592)
    g = (99.4708025861 * math.log(t) - 161.1195681661) if t <= 66.0 \
        else 288.1221695283 * ((t - 60.0) ** -0.0755148492)
    if t >= 66.0:
        b = 255.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10.0) - 305.0447927307
    return tuple(max(0.0, min(1.0, v / 255.0)) for v in (r, g, b))


worst = 0.0
for T in (1000, 2500, 4000, 6500, 8000, 10000, 20000, 40000):
    a = R.blackbody_rgb(T)
    b = blackbody_glsl(T)
    worst = max(worst, max(abs(a[i] - b[i]) for i in range(3)))
ok('八个温度下两种实现一致（差 < 1e-12）', worst < 1e-12, '最大差 %.3e' % worst)

# ── 6. 星场 ───────────────────────────────────────────────────────────────
print('')
print('=== 6. 星场：哈希常数与角度度量 ===')
for const in ('73856093', '19349663', '83492791', '1274126177'):
    ok('哈希常数 %s 在 GLSL 里' % const, const in CODE)
ok('用叉积模长当角度度量',
   re.search(r'length\s*\(\s*cross\s*\(\s*d\s*,\s*sd\s*\)\s*\)', CODE) is not None)
ok('星点核 exp(-sin²θ/(2σ²))',
   re.search(r'exp\s*\(\s*-\s*\(\s*sinTheta\s*\*\s*sinTheta\s*\)\s*/\s*'
             r'\(\s*2\.0\s*\*\s*sigma\s*\*\s*sigma\s*\)\s*\)', CODE) is not None)
ok('星点亮度用 amp³（帕累托分布）',
   re.search(r'amp\s*=\s*amp\s*\*\s*amp\s*\*\s*amp', CODE) is not None)

glsl_tiers = re.findall(r'layer\s*==\s*0\)\s*\?\s*([0-9.]+)\s*:\s*([0-9.]+)', CODE)
ok('两层星参数是成对三元表达式（至少 4 对）', len(glsl_tiers) >= 4, '%d 对' % len(glsl_tiers))

freqs = re.findall(r'sin\s*\(\s*d\.[xyz]\s*\*\s*([0-9.]+)', CODE)
hi = [f for f in freqs if float(f) > 4.0]
ok('银河带正弦频率都 ≤ 4（低频，不会变斑块）', not hi,
   '频率 ' + (', '.join(freqs) if freqs else '没找到'))

# ── 7. 相机基 ─────────────────────────────────────────────────────────────
print('')
print('=== 7. 相机基：JS 与 CPU 版一致 ===')


def camera_basis_cpu(dist, elev_deg, azim_deg=0.0):
    e = math.radians(elev_deg)
    a = math.radians(azim_deg)
    pos = np.array([dist * math.cos(e) * math.cos(a), dist * math.sin(e),
                    dist * math.cos(e) * math.sin(a)])
    forward = -pos / np.linalg.norm(pos)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return pos, forward, right, up


def camera_basis_js(dist, elev_deg):
    """逐行照抄 web/index.html 里的 cameraBasis()。"""
    e = elev_deg * math.pi / 180
    pos = [dist * math.cos(e), dist * math.sin(e), 0]
    L = math.hypot(*pos)
    fwd = [-pos[0] / L, -pos[1] / L, -pos[2] / L]
    WU = [0, 1, 0]
    right = [fwd[1] * WU[2] - fwd[2] * WU[1],
             fwd[2] * WU[0] - fwd[0] * WU[2],
             fwd[0] * WU[1] - fwd[1] * WU[0]]
    rl = math.hypot(*right)
    right = [1, 0, 0] if rl < 1e-9 else [c / rl for c in right]
    up = [right[1] * fwd[2] - right[2] * fwd[1],
          right[2] * fwd[0] - right[0] * fwd[2],
          right[0] * fwd[1] - right[1] * fwd[0]]
    return pos, fwd, right, up


worst = 0.0
for dist, elev in ((40.0, 9.0), (10.0, 0.0), (60.0, -35.0), (25.0, 60.0)):
    pc, fc, rc, uc = camera_basis_cpu(dist, elev)
    pj, fj, rj, uj = camera_basis_js(dist, elev)
    worst = max(worst,
                max(abs(pc[i] - pj[i]) for i in range(3)),
                max(abs(fc[i] - fj[i]) for i in range(3)),
                max(abs(rc[i] - rj[i]) for i in range(3)),
                max(abs(uc[i] - uj[i]) for i in range(3)))
ok('四组相机参数下 JS 与 CPU 的基向量一致（差 < 1e-12）', worst < 1e-12,
   '最大差 %.3e' % worst)

pc, fc, rc, uc = camera_basis_cpu(40.0, 9.0)
dev = float(np.linalg.norm(np.cross(rc, uc) + fc))
ok('right × up = -forward（手性一致）', dev < 1e-12, '偏差 %.2e' % dev)

# ── 8. 宿主默认值与 CPU 版对齐 ────────────────────────────────────────────
print('')
print('=== 8. 滑块默认值 vs CPU 版默认值 ===')


def slider_value(sid):
    mm = re.search(r'id="' + sid + r'"[^>]*value="([-0-9.]+)"', html)
    return float(mm.group(1)) if mm else None


for sid, expected, label in (
        ('douter', R.DISK_OUTER, '盘外缘'),
        ('dop', R.DISK_OPACITY, '盘不透明度'),
        ('dtemp', R.DISK_T_ISCO, '内缘温度'),
        ('dist', R.CAM_DIST, '相机距离'),
        ('elev', R.CAM_ELEV_DEG, '相机仰角'),
        ('fov', R.FOV_Y, '视场角')):
    v = slider_value(sid)
    ok('滑块 %s (%s) 默认 %.2f' % (sid, label, expected),
       v is not None and abs(v - expected) < 1e-9,
       ('实际 ' + str(v)) if v is not None else '没找到滑块')

ok('角步长换算 ÷100 覆盖验证过的 0.02',
   re.search(r'uDphi,\s*parseInt\(P\.dphi\.value,\s*10\)\s*/\s*100\.0', html) is not None)

# ── 9. 安全约束 ───────────────────────────────────────────────────────────
print('')
print('=== 9. 安全约束在 GLSL 里真的生效 ===')
ok('MAX_STEPS 是 #define 常量', re.search(r'#define\s+MAX_STEPS\s+\d+', CODE) is not None)
ok('MAX_DISK_HITS 是 #define 常量', re.search(r'#define\s+MAX_DISK_HITS\s+\d+', CODE) is not None)
ok('主循环上界是 MAX_STEPS',
   re.search(r'for\s*\(\s*int\s+i\s*=\s*0\s*;\s*i\s*<\s*MAX_STEPS', CODE) is not None)
ok('循环内用 uniform 提前收尾（i >= uSteps）',
   re.search(r'i\s*>=\s*uSteps\s*\)\s*break', CODE) is not None)
ok('有抖动让相邻像素步长不同',
   'dither' in CODE and re.search(r'dphi\s*=\s*uDphi\s*\*\s*\(\s*1\.0\s*\+\s*dither', CODE) is not None)
ok('盘采样有次数上限', re.search(r'diskHits\s*<\s*MAX_DISK_HITS', CODE) is not None)
ok('宿主侧有看门狗', re.search(r'WATCHDOG_MS\s*=\s*\d+', html) is not None)
ok('宿主侧有分辨率缩放', 'scalePct' in html)

print('')
print('=' * 66)
print('通过 %d 项，失败 %d 项' % (passed, failed))
if failed:
    print('')
    for f in failures:
        print('  失败: ' + f)
print('')
print('注意：本脚本验证的是「GLSL 文本与已验证的 CPU 版数学一致」，')
print('      它不能替代真实编译与运行 —— 驱动层面仍以浏览器为准。')
sys.exit(0 if failed == 0 else 1)
