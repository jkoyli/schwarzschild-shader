"""
数学自检 —— 在渲染任何像素之前，先证明史瓦西光线追踪是对的。

═══════════════════════════════════════════════════════════════════════════════
为什么要先跑这个（第一版就是在这里栽的）
═══════════════════════════════════════════════════════════════════════════════
黑洞的视觉特征（黑盘、光环、星空扭曲）对常数**并不敏感** —— 常数写错 30%，
画面依然"看起来很像黑洞"。所以肉眼判断是靠不住的，必须有数值判据。

第一版照抄了网上那个"等价牛顿力场 F = -(3/2)h²/r⁵"的说法，在笛卡尔坐标下积分，
结果偏折角比理论小 **1200 倍**。如果直接去渲染，只会得到一张"星空稍微有点歪"的图，
根本不会有黑洞的样子 —— 而那时我会以为是参数没调好，去调亮度、调相机距离，
白烧一整天。这就是先做数值自检的价值。

判据 A（弱场偏折）：α = 4GM/(c²b) = 2·r_s/b。
    这是广义相对论的经典结果（也是 1919 年日食验证的那个）。
    它同时锁定「方程 (1) 里的 3/2 常数」和「RK4 积分器」两件事：
    常数若写成 1.0 会得到 4/3·(r_s/b)，写成 2.0 会得到 8/3·(r_s/b)。

判据 B（临界碰撞参数）：b_c = 3√3/2 ≈ 2.598076。
    用二分法找"落入 / 逃逸"的分界点，它必须等于解析值 ——
    这条直接决定黑洞阴影在画面上有多大。
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schwarzschild import (  # noqa: E402
    RS, PHOTON_SPHERE, SHADOW_RADIUS, ISCO,
    critical_impact_parameter, trace_geodesic, deflection_angle,
)

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


def near(a, b, tol):
    return abs(a - b) <= tol


print('')
print('=== 0. 单位制与解析常量 ===')
ok('事件视界 r_s = 1', near(RS, 1.0, 1e-12))
ok('光子球 = 1.5 r_s', near(PHOTON_SPHERE, 1.5, 1e-12))
ok('ISCO = 3 r_s', near(ISCO, 3.0, 1e-12))
ok('临界碰撞参数 b_c = 3sqrt(3)/2 = %.12f' % critical_impact_parameter(),
   near(critical_impact_parameter(), 2.598076211353316, 1e-12))
ok('b_c 与 1.5*sqrt(3) 一致',
   near(SHADOW_RADIUS, 1.5 * math.sqrt(3.0), 1e-15))

print('')
print('=== 1. 判据 A：弱场偏折的**收敛阶**必须符合 2*r_s/b + O(1/b^2) ===')
# 【为什么不直接断言"比值≈1"】
# 弱场公式 α = 2r_s/b 只是领头阶，真实展开是 α = 2/b + c₂/b² + …
# 拿 b=30 去比就会差 5%，那不是代码错，是物理 —— 第一版我把容差放到 5% 就想蒙过去，
# 结果 b=30 差 5.2% 正好卡在边界上。正确做法是检验**收敛行为**：
# 残差 r(b) ≡ α(b) - 2/b 应当 ∝ 1/b²，于是 r(b)/r(2b) → 4。
# 这条比"比值≈1"强得多：它同时验证了正确的领头项**和**正确的次领头项幂次。
resid = []
for b in (60.0, 120.0, 240.0, 480.0):
    alpha = deflection_angle(b, dphi=0.001)
    r = alpha - 2.0 / b
    resid.append((b, alpha, r))
    ok('b = %-6g  α=%.8e   残差 r=α-2/b=%.6e' % (b, alpha, r), r > 0,
       '(残差应为正)')

ratios = [resid[i][2] / resid[i + 1][2] for i in range(len(resid) - 1)]
ok('残差比 r(b)/r(2b) -> 4（证明展开式是 2/b + O(1/b^2)）',
   all(3.6 < x < 4.4 for x in ratios),
   ' / '.join('%.3f' % x for x in ratios))

# 反过来：如果常数写错（1.0 或 2.0），残差相对领头项的比例会变成常数而非衰减
lead_ratio = [resid[i][2] / (2.0 / resid[i][0]) for i in range(len(resid))]
ok('残差相对领头项单调衰减（排除"常数写错"的情形）',
   all(lead_ratio[i] > lead_ratio[i + 1] for i in range(len(lead_ratio) - 1)),
   ' / '.join('%.5f' % x for x in lead_ratio))

print('')
print('=== 2. 判据 B：二分找临界碰撞参数，须 = 2.598076 ===')


def falls_in(b):
    return trace_geodesic(b, dphi=0.002).status == 'horizon'


lo, hi = 1.0, 12.0
if not falls_in(lo):
    ok('b=%g 应被捕获' % lo, False, '却逃逸了 —— 方程或常数有问题')
elif falls_in(hi):
    ok('b=%g 应逃逸' % hi, False, '却被捕获 —— 方程或常数有问题')
else:
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if falls_in(mid):
            lo = mid
        else:
            hi = mid
    bc = 0.5 * (lo + hi)
    rel = abs(bc - SHADOW_RADIUS) / SHADOW_RADIUS
    ok('数值 b_c = %.6f' % bc, rel < 0.005,
       '解析 %.6f，相对误差 %.4f%%' % (SHADOW_RADIUS, rel * 100))

print('')
print('=== 3. 光子球附近的行为（决定光环细节） ===')
# 越靠近 b_c，光线在光子球附近停留越久 → phi_total 应当**单调增长**。
# 这条比"必须绕 N 圈"更稳：绕圈数对 b-b_c 是**对数发散**的，
# 定死一个圈数是错的判据（第一版就是这么写错的）。
windings = []
for factor in (1.1, 1.02, 1.005, 1.0015, 1.0005):
    res = trace_geodesic(SHADOW_RADIUS * factor, dphi=0.001)
    windings.append((factor, res.status, res.phi_total))
    print('       b = %.6f × b_c   status=%-8s phi_total = %7.3f rad (%.2f 圈)'
          % (factor, res.status, res.phi_total, res.phi_total / (2 * math.pi)))

ok('全部逃逸（都在 b_c 之外）',
   all(w[1] == 'escaped' for w in windings))
ok('绕圈随 b 逼近 b_c 而单调增长',
   all(windings[i][2] < windings[i + 1][2] for i in range(len(windings) - 1)),
   ' / '.join('%.2f' % w[2] for w in windings))

# 【判据：对数发散本身，而不是某个圈数阈值】
# 理论上 φ_total(b) ≈ -ln(b/b_c - 1) + C。等价的可测形式是：
# 当 (b/b_c - 1) 每次**减半**时，φ_total 应当每次增加 ln2 ≈ 0.693。
# 第一版我写的是"必须绕 2 圈以上"，那是个任意阈值 —— 绕多少圈取决于离 b_c 多近，
# 而定死这个数只会逼着自己把 dphi 越调越小，是错的判据。
half_steps = []
for i in range(len(windings) - 1):
    x0 = windings[i][0] - 1.0
    x1 = windings[i + 1][0] - 1.0
    if x0 > 0 and x1 > 0:
        ratio = x0 / x1
        dphi_inc = windings[i + 1][2] - windings[i][2]
        # 归一化到"每次减半"的等效增量
        expected = math.log(2.0)
        scaled = dphi_inc * (math.log(2.0) / math.log(ratio)) if ratio > 1 else 0.0
        half_steps.append((ratio, dphi_inc, scaled))

for ratio, inc, scaled in half_steps:
    print('       区间比 %.2f×   Δφ=%.3f rad   折算到"减半" = %.3f rad' % (ratio, inc, scaled))
ok('每减半一次，φ_total 增加约 ln2=0.693（对数发散规律）',
   all(0.25 < s < 1.6 for _, _, s in half_steps),
   '折算值 ' + ' / '.join('%.3f' % s for _, _, s in half_steps))
ok('最接近 b_c 的那条至少绕了 1.5 圈', windings[-1][2] > 3.0 * math.pi,
   '%.2f 圈' % (windings[-1][2] / (2 * math.pi)))

# 远场应当几乎不绕：phi_total ≈ pi
far = trace_geodesic(200.0, dphi=0.002)
ok('b = 200 的光线几乎不绕（phi_total 接近 pi）',
   far.status == 'escaped' and near(far.phi_total, math.pi, 0.05),
   'phi_total = %.6f, pi = %.6f' % (far.phi_total, math.pi))

print('')
print('=== 4. 数值稳定性：步长不改变结论 ===')
vals = []
for dphi in (0.02, 0.01, 0.005, 0.0025):
    a = deflection_angle(5.0, dphi=dphi)
    vals.append((dphi, a))
spread = max(v for _, v in vals) - min(v for _, v in vals)
ok('四种步长下偏折角一致（相对差 < 1%）', spread / vals[-1][1] < 0.01,
   ' / '.join('dphi=%.4f: %.6f' % (d, a) for d, a in vals))

print('')
print('=== 5. 单调性：b 越大偏折越小 ===')
seq = [deflection_angle(b) for b in (3.0, 5.0, 10.0, 30.0)]
ok('偏折角随 b 单调递减',
   all(seq[i] > seq[i + 1] for i in range(len(seq) - 1)),
   ' / '.join('%.5f' % s for s in seq))

print('')
print('-' * 64)
print('通过 %d 项，失败 %d 项' % (passed, failed))
if failed:
    print('')
    print('数学层没通过，先别渲染 —— 渲染出来的"好看"会掩盖错误。')
    for f in failures:
        print('  失败: ' + f)
sys.exit(0 if failed == 0 else 1)
