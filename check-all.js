/**
 * 四层验证，一次跑完。
 *
 * 用法: node check-all.js
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 为什么要四层（每层抓的是不同类型的错，缺一层就有盲区）
 * ═══════════════════════════════════════════════════════════════════════════
 *   1. 数学自检      —— 物理学对不对（弱场偏折收敛阶 / 临界碰撞参数 / 光子球对数发散）
 *   2. GLSL 静态校验 —— 语法层：禁用标识符 / 循环上界 / uniform 上传是否配套
 *   3. 一致性审计    —— 两份实现（Python 与 GLSL）是不是同一个数学
 *   4. 无头软件渲染  —— 真的过一遍 GLSL 编译器并渲染几帧（SwiftShader，不碰真实 GPU）
 *
 * 【为什么不用 .cmd 写这个】
 * 上一轮用 .cmd 写过一个类似的运行脚本，中文注释在 cmd 里按 GBK 解析，
 * 整份脚本被拆成乱码命令、退出码还是 0 —— 一个"看起来跑过了其实什么都没跑"的假通过。
 * 用 Node 没这个问题。
 */
'use strict';
const { spawnSync } = require('child_process');
const path = require('path');

const ROOT = __dirname;

const STEPS = [
  { title: '1/4 数学自检（CPU，物理判据）', cwd: path.join(ROOT, 'cpu'), cmd: 'test_math.py', kind: 'py' },
  { title: '2/4 GLSL 静态校验（语法 / 安全约束）', cwd: path.join(ROOT, 'web'), cmd: 'glsl_lint.js', kind: 'node' },
  { title: '3/4 GLSL-Python 一致性审计', cwd: path.join(ROOT, 'cpu'), cmd: 'parity_check.py', kind: 'py' },
  { title: '4/4 无头软件渲染（真实编译 + 渲几帧）', cwd: path.join(ROOT, 'web'), cmd: 'headless_verify.js', kind: 'node' },
];

const HINTS = [
  '数学层没过 —— 先别渲染，渲染出来的"好看"会掩盖错误。',
  'GLSL 语法/安全约束没过 —— 有会挂驱动的写法，必须先修。',
  '两份实现的数学不一致 —— 系数手滑了，画面会"稍微不对"且没有任何报错。',
  '着色器编译或运行失败 —— 报错文本里带行号，对照 web/index.html 里的 GLSL 改。',
];

let failedAt = -1;

for (let i = 0; i < STEPS.length; i++) {
  const s = STEPS[i];
  console.log('');
  console.log('='.repeat(64));
  console.log(s.title);
  console.log('='.repeat(64));

  const args = s.kind === 'py' ? [s.cmd] : [s.cmd];
  const bin = s.kind === 'py' ? 'python' : process.execPath;

  const r = spawnSync(bin, args, { cwd: s.cwd, stdio: 'inherit', shell: false });

  if (r.error) {
    console.log('启动失败: ' + String(r.error.message));
    failedAt = i;
    break;
  }
  if (r.status !== 0) {
    failedAt = i;
    break;
  }
}

console.log('');
console.log('='.repeat(64));
if (failedAt < 0) {
  console.log('四层全部通过。');
  console.log('');
  console.log('  CPU 参考渲染器 :  cpu/render.py  然后 cpu/compose.py');
  console.log('  GPU 实时版     :  浏览器打开 web/index.html');
} else {
  console.log('在第 ' + (failedAt + 1) + ' 层失败。');
  console.log('');
  console.log('  ' + HINTS[failedAt]);
}
console.log('='.repeat(64));

process.exit(failedAt < 0 ? 0 : 1);
