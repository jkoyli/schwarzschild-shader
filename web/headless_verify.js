/**
 * 无 GPU 风险的着色器验证 —— 用 Edge 的无头模式 + SwiftShader 软件渲染
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 为什么这样设计（两个目标同时达成）
 * ═══════════════════════════════════════════════════════════════════════════
 * 1. **真的过一遍 GLSL 编译器** —— 静态检查再全，也替代不了驱动/ANGLE 的编译。
 *    ANGLE（Chromium 的 GL 实现层）会真编译我们的 GLSL ES 3.00。
 *
 * 2. **完全不碰真实 GPU** —— 加 `--use-gl=swiftshader` 后，渲染全部在 CPU 上
 *    用软件光栅化完成。**驱动超时（TDR）的前提是"GPU 被命令卡住"，
 *    软件渲染根本不会发生这件事。** 所以这个检查对用户的电脑零风险，
 *    这正是"要先有真实沙箱"这条要求的落地方式。
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 它验证什么、不验证什么
 * ═══════════════════════════════════════════════════════════════════════════
 * 验证：
 *   · GLSL 能不能编译（语法、类型、精度限定符、循环上界规则）
 *   · 能不能链接（in/out 配对、uniform 名字）
 *   · 首帧渲染路径能不能跑完不抛 JS 异常（WebGL 调用合法性）
 *   · 页面的自检输出（shader 报错文本、GPU 字符串、FPS）
 *
 * 不验证：
 *   · 画面好不好看（SwiftShader 慢，而且不做像素级比对）
 *   · 真实 GPU 驱动上的行为差异
 *
 * 用法: node headless_verify.js
 */
'use strict';
const { execFileSync, spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const here = __dirname;
const indexPath = path.join(here, 'index.html');
const selfcheckPath = path.join(here, 'selfcheck.html');

let pass = 0, fail = 0;
const failures = [];
function ok(name, cond, detail) {
  if (cond) { pass++; console.log('  [OK]   ' + name + (detail ? '  ' + detail : '')); }
  else { fail++; failures.push(name); console.log('  [FAIL] ' + name + (detail ? '  ' + detail : '')); }
}

// ── 从 index.html 抽出真实着色器源码 ──────────────────────────────────────
// 【为什么要注入而不是复制一份】
// 两份拷贝一定会不同步 —— 改了 index.html 忘了改自检页，
// 自检就会"通过"一个已经不存在的版本。所以自检页只留占位符，
// 由本脚本在运行时把**当前**源码塞进去。
function extractShaders(html) {
  const out = {};
  const re = /const\s+([A-Z_][A-Z0-9_]*)\s*=\s*`([\s\S]*?)`;/g;
  let m;
  while ((m = re.exec(html)) !== null) {
    if (/#version/.test(m[2])) out[m[1]] = m[2];
  }
  return out;
}

const indexHtml = fs.readFileSync(indexPath, 'utf8');
const shaders = extractShaders(indexHtml);
console.log('');
console.log('='.repeat(66));
console.log('无头软件渲染着色器验证');
console.log('='.repeat(66));
console.log('从 index.html 抽到着色器: ' + Object.keys(shaders).join(', '));
ok('抽到了 VS / FS_MAIN / FS_POST',
   !!shaders.VS && !!shaders.FS_MAIN && !!shaders.FS_POST);

// 把源码注入自检页，写到临时文件（不污染项目目录）
//
// ⚠ 【一个很阴的坑】这里必须用**函数式替换**，不能用 String.replace(字符串, ...)。
// 后者只替换**第一次出现**，而占位符名字在注释里也出现过一次 ——
// 结果 JSON 被塞进了注释、代码里仍是不定义的标识符，
// 页面报 "Uncaught ReferenceError: ... is not defined"，
// 而整个流程看起来"注入成功了"（因为确实替换了，只是替换错了地方）。
// 用函数式替换可以一次替换全部，并且能断言替换次数。
const PLACEHOLDER = '__SHADER' + '_SOURCE__';
const tmpDir = path.join(os.tmpdir(), 'bh_selfcheck_' + Date.now());
fs.mkdirSync(tmpDir, { recursive: true });
const tmpPage = path.join(tmpDir, 'selfcheck.html');

const selfcheckSrc = fs.readFileSync(selfcheckPath, 'utf8');
let replaceCount = 0;
const injected = selfcheckSrc.replace(new RegExp(PLACEHOLDER, 'g'), () => {
  replaceCount++;
  return JSON.stringify(shaders);
});
ok('占位符被替换了（且只应有代码里那一处，除非注释也写了）', replaceCount >= 1,
   '替换 ' + replaceCount + ' 处');
fs.writeFileSync(tmpPage, injected, 'utf8');
console.log('自检页已生成: ' + tmpPage + '  (' + injected.length + ' 字节)');

// ── 找 Edge / Chrome ─────────────────────────────────────────────────────
function findBrowser() {
  const cands = [
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
  ];
  for (const c of cands) if (fs.existsSync(c)) return c;
  return null;
}

const browser = findBrowser();
console.log('浏览器: ' + (browser || '（没找到 Edge / Chrome）'));
ok('找到可用的 Chromium 系浏览器', browser !== null);

if (!browser) {
  console.log('');
  console.log('没找到浏览器，跳过这一步。');
  console.log('（静态检查仍然有效，见 glsl_lint.js / parity_check.py）');
  try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch (e) { /* ignore */ }
  process.exit(1);
}

// ── 跑无头浏览器 ─────────────────────────────────────────────────────────
// 关键参数说明：
//   --headless=new          新版无头模式
//   --use-gl=swiftshader    ★ 纯软件渲染，不碰真实 GPU（零 TDR 风险）
//   --use-angle=swiftshader 同上，ANGLE 层也走软件
//   --disable-gpu           不再尝试硬件加速
//   --dump-dom              把最终 DOM 打印到 stdout（自检结果就写在里面）
//   --virtual-time-budget   给页面足够的虚拟时间跑完 WebGL 初始化
const tmpProfile = path.join(os.tmpdir(), 'bh_headless_' + Date.now());
const args = [
  '--headless=new',
  '--disable-gpu',
  '--use-gl=swiftshader',
  '--use-angle=swiftshader',
  '--enable-unsafe-swiftshader',
  '--no-sandbox',
  '--disable-extensions',
  '--virtual-time-budget=15000',
  '--user-data-dir=' + tmpProfile,
  '--dump-dom',
  'file:///' + tmpPage.replace(/\\/g, '/'),
];

console.log('');
console.log('启动无头浏览器（软件渲染，不占用真实 GPU）…');
let dom = '';
let runErr = null;
try {
  dom = execFileSync(browser, args, {
    encoding: 'utf8',
    timeout: 90000,
    maxBuffer: 64 * 1024 * 1024,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
} catch (e) {
  runErr = e;
  dom = String((e && e.stdout) || '');
}

ok('无头浏览器跑完（没有超时/崩溃）', runErr === null,
   runErr ? String(runErr.message).slice(0, 200) : (dom.length + ' 字节 DOM'));

// ── 从 DOM 里抠出自检结果 ────────────────────────────────────────────────
// 页面把结果写成 <div id="result" data-status="...">JSON</div>
const m = /<div id="result"[^>]*data-status="([^"]*)"[^>]*>([\s\S]*?)<\/div>/.exec(dom);
if (!m) {
  console.log('');
  console.log('没能在 DOM 里找到自检结果块。');
  console.log('原始 DOM 片段（前 1200 字节）：');
  console.log(dom.slice(0, 1200));
  ok('页面输出了自检结果', false);
} else {
  const status = m[1];
  let payload = null;
  try {
    payload = JSON.parse(m[2].replace(/&quot;/g, '"').replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<').replace(/&gt;/g, '>'));
  } catch (e) {
    console.log('结果块 JSON 解析失败: ' + String(e.message));
    console.log(m[2].slice(0, 800));
  }

  console.log('');
  console.log('--- 页面自检结果 ---');
  ok('页面自检状态 = ok', status === 'ok', 'status=' + status);

  if (payload) {
    ok('拿到了 WebGL2 上下文', payload.hasContext === true,
       'renderer=' + String(payload.renderer).slice(0, 70));
    ok('主着色器编译成功', payload.mainCompiled === true,
       payload.mainError ? String(payload.mainError).slice(0, 400) : '');
    ok('后处理着色器编译成功', payload.postCompiled === true,
       payload.postError ? String(payload.postError).slice(0, 400) : '');
    ok('程序链接成功', payload.mainLinked === true && payload.postLinked === true);
    ok('浮点帧缓冲可用（RGBA16F）', payload.fboOk === true,
       payload.fboError || '');
    ok('连续渲染 10 帧没有 WebGL 错误', payload.framesOk === true,
       'glError=' + String(payload.glError));
    ok('没有 JS 运行时异常', payload.jsError === null,
       String(payload.jsError || '').slice(0, 300));

    // 输出画面统计，确认"确实画出东西了"而不是全黑
    if (payload.pixelStats) {
      const s = payload.pixelStats;
      ok('场景 pass 不是全黑', s.max > 1,
         'min=' + s.min + ' max=' + s.max + ' mean=' + s.mean);
    }
    // 把最终上屏画面存成 PNG —— 这样在无 GPU 的机器上也能"亲眼看一眼"
    if (payload.screenPng && /^data:image\/png;base64,/.test(payload.screenPng)) {
      const b64 = payload.screenPng.split(',')[1];
      const outDir = path.join(here, 'out');
      fs.mkdirSync(outDir, { recursive: true });
      const pngPath = path.join(outDir, 'gpu_swiftshader.png');
      fs.writeFileSync(pngPath, Buffer.from(b64, 'base64'));
      const sz = fs.statSync(pngPath).size;
      ok('已把 GPU 最终画面存成 PNG', sz > 100,
         path.basename(pngPath) + '  ' + sz + ' 字节  平均亮度 ' + payload.screenMean);
    } else {
      ok('拿到了上屏画面的 PNG', false, '没拿到 screenPng');
    }

    console.log('');
    console.log('--- 详细输出 ---');
    console.log('GPU 字符串 : ' + payload.renderer);
    console.log('GLSL 版本  : ' + payload.glslVersion);
    console.log('场景 pass  : min=' + (payload.pixelStats ? payload.pixelStats.min : '?')
      + ' max=' + (payload.pixelStats ? payload.pixelStats.max : '?')
      + ' mean=' + (payload.pixelStats ? payload.pixelStats.mean : '?'));
    console.log('上屏画面   : ' + payload.screenSize + '  平均亮度 ' + payload.screenMean);
    if (payload.log) console.log('页面日志   : ' + payload.log);
  }
}

// ── 清理临时文件 ─────────────────────────────────────────────────────────
try {
  fs.rmSync(tmpProfile, { recursive: true, force: true });
} catch (e) { /* 忽略 */ }
try {
  fs.rmSync(tmpDir, { recursive: true, force: true });
} catch (e) { /* 忽略 */ }

console.log('');
console.log('='.repeat(66));
console.log('通过 ' + pass + ' 项，失败 ' + fail + ' 项');
if (fail) {
  console.log('');
  for (const f of failures) console.log('  失败: ' + f);
  console.log('');
  console.log('提示：着色器报错文本里带行号，对照 web/index.html 里的 GLSL 改即可。');
}
console.log('');
console.log('说明：本次用的是 SwiftShader **软件渲染**，不占用真实 GPU，');
console.log('      因此不存在驱动超时(TDR)风险。真实 GPU 上的表现仍需你在浏览器里确认。');
process.exit(fail === 0 ? 0 : 1);
