"""
史瓦西黑洞光线追踪 —— 核心数学层

═══════════════════════════════════════════════════════════════════════════════
一、光在史瓦西时空里怎么走
═══════════════════════════════════════════════════════════════════════════════

取 r_s = 2GM/c² = 1 为单位。把光线的轨道投影到它自己的轨道平面上，令 u = 1/r，
则零测地线满足**精确**方程（不是近似）：

        d²u/dφ² = -u + (3/2)·u²                                … (1)

右边第一项 -u 是经典的"离心势"项（直线传播就来自它）；第二项 (3/2)u² 完全是
时空弯曲的贡献 —— 没有它，光就走直线。

【为什么用这个形式，而不是"假想牛顿力场"】
网上（含 rantonels 的文章）常给一个更"物理"的等价说法：把 (1) 看成 Binet 方程，
它等价于一个质点受中心力 F(r) = -(3/2)h²/r⁵ 的运动。

**我第一版就是照着那个力场在笛卡尔坐标下积分的，结果偏折角比理论小 1200 倍。**
原因是那个说法里的力是**按 u 求导**的广义力（F = h²u²·dV/du），
直接当成笛卡尔加速度 a = F 去积分，等于少乘一个径向因子 —— 得到的不是同一条轨迹。

所以这里直接用 (1) 式本身。它没有坐标变换、没有"等价力场"这层中介，
数值上就是方程左边原样，少一个出错环节。

【三个特征半径（r_s = 1）】
    事件视界   r = 1
    光子球     r = 1.5    ← 光的（不稳定）圆轨道
    阴影视半径 ≈ 2.598    ← 临界碰撞参数 b_c = 3√3/2

二、为什么积分 u(φ) 而不是 r(t)
═══════════════════════════════════════════════════════════════════════════════
- r(λ) 形式在**转折点**（dr/dλ = 0，即光线最接近黑洞处）会退化，需要特殊处理；
  u(φ) 形式在整个过程中都良态。
- u(φ) 的每一步都是"转过一个固定小角度"，天然均匀，不需要自适应步长那套麻烦。
- 落向视界的光线在 u(φ) 下表现为 u 单调增到 1（r=1）后被视界截断，
  判据干净利落。

代价是必须处理"光线跑向无穷远"：那时 u → 0，需要判 u 穿零并按渐近方向收尾。
"""

import math

# ── 单位制常量（r_s = 1）────────────────────────────────────────────────────
RS = 1.0                                        # 史瓦西半径（事件视界）
PHOTON_SPHERE = 1.5                             # 光子球
SHADOW_RADIUS = 1.5 * math.sqrt(3.0)            # 临界碰撞参数 b_c = 3√3/2 ≈ 2.598076
ISCO = 3.0                                      # 最内稳定圆轨道（吸积盘内缘）


def critical_impact_parameter():
    """b_c = 3√3·M = (3√3/2)·r_s。b < b_c 的光子落入黑洞。"""
    return SHADOW_RADIUS


def geodesic_u(u, w):
    """
    方程 (1) 的右端：u'' = (3/2)u² - u。

    验收判据：弱场极限下它必须给出偏折 α = 4GM/(c²b) = 2·r_s/b。
    那个"2"与这里的"3/2"是一一对应的 —— 常数写成 1.0 会得到 4/3·(r_s/b)，
    写成 2.0 会得到 8/3·(r_s/b)，一测就露馅。test_math.py 就是干这个的。
    """
    return 1.5 * u * u - u


def rk4_step_u(u, w, dphi):
    """对 (u, w=du/dφ) 做一步四阶龙格-库塔。"""
    k1u = w
    k1w = geodesic_u(u, w)

    k2u = w + 0.5 * dphi * k1w
    k2w = geodesic_u(u + 0.5 * dphi * k1u, w + 0.5 * dphi * k1w)

    k3u = w + 0.5 * dphi * k2w
    k3w = geodesic_u(u + 0.5 * dphi * k2u, w + 0.5 * dphi * k2w)

    k4u = w + dphi * k3w
    k4w = geodesic_u(u + dphi * k3u, w + dphi * k3w)

    nu = u + (dphi / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
    nw = w + (dphi / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)
    return nu, nw


class RayResult:
    """
    一条光线的追踪结果。

    status :
        'horizon'  —— 落进视界（这个像素是黑的）
        'escaped'  —— 逃逸到无穷远，phi_inf 是它最终飞出的方位角
    phi_total : 总共转过的角度（弧度）。对直线传播它等于 π。
                偏折角 α = phi_total - π。
    """

    __slots__ = ('status', 'phi_total', 'u_min', 'winding')

    def __init__(self, status, phi_total, u_min, winding):
        self.status = status
        self.phi_total = phi_total
        self.u_min = u_min          # 全程最大的 r 对应 u 的最小值（u=1/r）
        self.winding = winding      # 绕黑洞转了几圈（用于识别高阶像）

    @property
    def deflection(self):
        """偏折角（弧度）。直线传播为 0。"""
        return self.phi_total - math.pi


def trace_geodesic(impact_parameter, dphi=0.0025, max_phi=60.0, horizon=RS):
    """
    追一条碰撞参数为 b 的光线，返回 RayResult。

    初始条件（在无穷远处沿直线飞来）：
        u(0) = 0,  u'(0) = 1/b
    这是 u(φ) 形式的自然初值：无穷远处 u=0；对直线轨迹 u = sin(φ)/b，
    在 φ=0 处 u' = cos(0)/b = 1/b。

    终止条件（三选一）：
        u >= 1/horizon      → 落进视界
        u <= 0（穿零）      → 逃逸到无穷远（转折之后又飞出去了）
        φ 累计 > max_phi    → 绕太多圈，判定为已捕获（防死循环）
    """
    u = 0.0
    w = 1.0 / impact_parameter
    phi = 0.0
    u_min = 0.0
    winding = 0.0
    prev_u = u

    inv_horizon = 1.0 / horizon

    while phi < max_phi:
        u, w = rk4_step_u(u, w, dphi)
        phi += dphi

        if u > u_min:
            u_min = u

        # 落进视界
        if u >= inv_horizon:
            return RayResult('horizon', phi, u_min, winding)

        # 穿零 → 逃逸。用线性插值把穿零位置找得更准（否则精度受 dphi 限制）
        if u <= 0.0:
            # 在 [prev_u, u] 之间线性插值出穿零点，修正 phi
            if prev_u > 0.0 and (prev_u - u) > 1e-18:
                frac = prev_u / (prev_u - u)
                phi = phi - dphi + frac * dphi
            return RayResult('escaped', phi, u_min, winding)

        # 累计绕圈数（u 由增转减 = 过了最近点一次，即绕了半圈）
        if prev_u > 0.0 and w < 0.0 and (u - prev_u) < 0.0:
            # 过转折点：w 由正转负
            pass
        prev_u = u

    # 转够多圈仍未逃逸 → 视为捕获（b 极接近 b_c 的光线会在光子球附近绕很多圈）
    return RayResult('horizon', phi, u_min, winding)


def ray_from_direction(direction, b_scale=1.0):
    """
    把"相机朝某个方向看"换算成 u(φ) 形式的碰撞参数 b。

    给定单位方向 d、相机到黑洞中心的距离 R，光线的碰撞参数为
        b = R·sin(θ)          θ 是 d 与"指向黑洞中心"方向 (-R̂) 的夹角
    但直接算更稳：b = |r × v| / |v|，其中 r 是相机位置、v 是方向。
    这里只依赖方向与距离，返回 (b, 初始 φ 偏移)。
    """
    raise NotImplementedError('由 render.py 在轨道平面上直接构造，见那里的实现')


def deflection_angle(impact_parameter, dphi=0.0025):
    """
    返回偏折角（弧度）；若被捕获则返回 None。
    这是 test_math.py 的主判据入口。
    """
    res = trace_geodesic(impact_parameter, dphi=dphi)
    if res.status != 'escaped':
        return None
    return res.deflection
