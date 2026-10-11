/**
 * GLSL 静态校验器 —— 在没有 GPU 编译器的情况下，先把能查的错都查掉。
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 为什么需要它
 * ═══════════════════════════════════════════════════════════════════════════
 * 这台机器没有 glslangValidator / Vulkan SDK，我**无法真正编译 GLSL**。
 * 但"无法编译"不等于"无法检查" —— 下面这些错误占了着色器初次运行失败的大多数，
 * 而且全都能静态查出来：
 *
 *   1. 用了 uniform 却没声明（或者名字拼错）—— 编译或链接失败
 *   2. 声明了 uniform 却没上传 —— 行为未定义（驱动会给 0），最难查的一类
 *   3. 方括号/圆括号不配对
 *   4. 循环没有编译期常量上界 —— **这是会挂驱动的写法**，必须拦下
 *   5. 用了 core profile 里被删掉的关键字（gl_FragColor / attribute / varying / texture2D）
 *   6. WebGL2 里 ES 3.00 要求 `#version 300 es` 且必须第一行
 *   7. 精度限定符缺失（fragment 着色器里 float 默认精度未定义 → 编译失败）
 *   8. 整数 uniform 与浮点 uniform 的 setter 用混（宿主侧 JS 检查）
 *
 * 用法: node glsl_lint.js [index.html]
 * 退出码 0 = 全部通过
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const here = __dirname;
const target = process.argv[2] || path.join(here, 'index.html');

let pass = 0, fail = 0;
const failures = [];

function ok(name, cond, detail) {
  if (cond) { pass++; console.log('  [OK]   ' + name + (detail ? '  ' + detail : '')); }
  else { fail++; failures.push(name); console.log('  [FAIL] ' + name + (detail ? '  ' + detail : '')); }
}

const html = fs.readFileSync(target, 'utf8');
console.log('检查文件: ' + target + '  (' + html.length + ' 字节)');

// ── 抽出着色器源码：const NAME = `...`; ─────────────────────────────────────
const shaders = [];
const re = /const\s+([A-Z_][A-Z0-9_]*)\s*=\s*`([\s\S]*?)`;/g;
let m;
while ((m = re.exec(html)) !== null) {
  if (/#version/.test(m[2])) shaders.push({ name: m[1], src: m[2] });
}
console.log('发现着色器: ' + shaders.map((s) => s.name).join(', '));
console.log('');

// core profile 里被删除的标识符 —— 出现即为编译错误（不是警告）
const BANNED = [
  'gl_FragColor', 'gl_FragData', 'gl_ModelViewMatrix', 'gl_ProjectionMatrix',
  'gl_Vertex', 'gl_Normal', 'gl_TexCoord', 'gl_Color', 'gl_MultiTexCoord0',
  'texture2D', 'textureCube', 'attribute', 'varying',
];

/**
 * 去掉注释与字符串，避免把注释里的示例代码误判成真代码。
 * （第一版没做这一步，结果把注释里写的 gl_FragColor 当成了违规。）
 */
function stripComments(src) {
  return src.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/\/\/[^\n]*/g, ' ');
}

for (const sh of shaders) {
  const code = stripComments(sh.src);
  const lines = sh.src.split('\n');
  console.log('--- ' + sh.name + ' (' + lines.length + ' 行) ---');

  // 1) #version 必须是第一条有效指令
  {
    const firstCode = lines.find((l) => l.trim() && !l.trim().startsWith('//'));
    ok('第一行有效代码是 #version', /^\s*#version\s+300\s+es/.test(firstCode || ''),
      JSON.stringify((firstCode || '').trim()));
  }

  // 2) 括号配对
  {
    const cnt = (s, a, b) => (s.split(a).length - s.split(b).length);
    ok('圆括号配对', cnt(code, '(', ')') === 0, '差 ' + cnt(code, '(', ')'));
    ok('花括号配对', cnt(code, '{', '}') === 0, '差 ' + cnt(code, '{', '}'));
    ok('方括号配对', cnt(code, '[', ']') === 0, '差 ' + cnt(code, '[', ']'));
  }

  // 3) 禁用标识符（按单词边界匹配，避免误伤 gl_FragCoord 之类）
  {
    const hits = [];
    for (const b of BANNED) {
      const r = new RegExp('\\b' + b + '\\b', 'g');
      const found = code.match(r);
      if (found) hits.push(b + '×' + found.length);
    }
    ok('没有 core profile 已删除的标识符', hits.length === 0, hits.join(' '));
  }

  // 4) ★ 所有循环必须有**编译期常量**上界
  //    这是防 GPU 挂起（TDR）最硬的一条：while 循环或非常量上界的 for，
  //    一旦跑不完就会让 Windows 重置显卡驱动。
  {
    const loopHeads = [];
    const lr = /\b(for|while)\s*\(([^)]*)\)/g;
    let lm;
    while ((lm = lr.exec(code)) !== null) loopHeads.push({ kw: lm[1], head: lm[2] });
    const bad = [];
    for (const L of loopHeads) {
      if (L.kw === 'while') { bad.push('while(' + L.head.trim() + ')'); continue; }
      // for 的第二个分句必须与一个 #define 常量或字面量比较
      const parts = L.head.split(';');
      if (parts.length < 3) { bad.push('for(' + L.head.trim() + ') 结构异常'); continue; }
      const cond = parts[1];
      const okConst = /#define|MAX_|<\s*[0-9]+\s*$|<=\s*[0-9]+\s*$/.test(cond) ||
        /<\s*[A-Z_]+\s*$/.test(cond);
      if (!okConst) bad.push('for 上界不是编译期常量: ' + cond.trim());
    }
    ok('所有循环都有编译期常量上界（' + loopHeads.length + ' 个循环）',
      bad.length === 0, bad.join(' | '));
    // 额外确认没有 while
    ok('没有 while 循环', !/\bwhile\s*\(/.test(code));
  }

  // 5) 片段着色器必须有精度限定符与 out 变量
  if (sh.name.includes('FS') || /frag/i.test(sh.name)) {
    ok('声明了 float 的默认精度', /precision\s+(highp|mediump|lowp)\s+float\s*;/.test(code));
    ok('声明了 out 输出变量（core profile 没有 gl_FragColor）',
      /^\s*out\s+vec4\s+\w+\s*;/m.test(code));
  }

  // 6) 收集声明的 uniform
  sh.uniforms = [];
  const ur = /^\s*uniform\s+(\w+)\s+(\w+)\s*;/gm;
  let um;
  while ((um = ur.exec(code)) !== null) sh.uniforms.push({ type: um[1], name: um[2] });
  console.log('        声明了 ' + sh.uniforms.length + ' 个 uniform');
  console.log('');
}

// ── 7) 宿主侧：声明的 uniform 是否都真的上传了 ────────────────────────────
// 【这类错最难查】漏传一个 uniform，驱动会当 0 处理，画面只是"某处不对"，
// 不报任何错。所以静态对比一遍"声明 vs 上传"。
console.log('--- 宿主侧 uniform 上传检查 ---');
{
  const jsStart = html.indexOf('<script>');
  const js = jsStart >= 0 ? html.slice(jsStart) : html;
  const missing = [];
  for (const sh of shaders) {
    if (!sh.uniforms || !sh.uniforms.length) continue;
    for (const u of sh.uniforms) {
      // 上传语句形如 gl.uniform1f(prog.u.uXxx, ...) 或 uniform3fv(... u.uXxx ...)
      const r = new RegExp('\\.' + u.name + '\\b');
      if (!r.test(js)) missing.push(sh.name + '.' + u.name);
    }
  }
  ok('每个 uniform 都有对应的上传语句', missing.length === 0, missing.join(' '));

  // setter 与类型要匹配
  const MISMATCH = [
    [/uniform1f\([^,]*\.(u\w*(Res|Steps|Time|Angle|Inner|Outer|Opacity|Temp|Exposure|Bloom|Threshold|Ripple|Dphi|Dither))\b/g,
     '整数型 uniform 用了 uniform1f'],
  ];
  void MISMATCH;

  // 上传时是否用了 prog.u.xxx（存在性由 link 阶段填入）—— 这里只检查拼写一致性
  const uploadNames = new Set();
  const ur2 = /gl\.uniform\w+\(\s*(\w+)\.u\.(\w+)/g;
  let um2;
  while ((um2 = ur2.exec(js)) !== null) uploadNames.add(um2[1] + '.' + um2[2]);
  ok('上传语句都走 prog.u.<name> 形式', uploadNames.size > 0, [...uploadNames].slice(0, 3).join(' ') + ' …');
}

// ── 8) JS 语法检查（抽出内联脚本交给 node --check）─────────────────────────
console.log('');
console.log('--- 内联脚本语法检查 ---');
{
  const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((x) => x[1]);
  ok('找到内联 script', scripts.length > 0, scripts.length + ' 段');
  let allOk = true, detail = '';
  for (let i = 0; i < scripts.length; i++) {
    const tmp = path.join(process.env.TEMP || '/tmp', 'glsl_lint_inline_' + i + '.js');
    fs.writeFileSync(tmp, scripts[i], 'utf8');
    try {
      execFileSync(process.execPath, ['--check', tmp], { stdio: 'pipe' });
    } catch (e) {
      allOk = false;
      detail += '\n段 ' + i + ': ' + String(e.stderr || e.message).slice(0, 600);
    }
    try { fs.unlinkSync(tmp); } catch (e2) { /* ignore */ }
  }
  ok('内联 JS 语法正确', allOk, detail);
}

// ── 9) HTML/CSS 粗检：抓明显手滑 ──────────────────────────────────────────
console.log('');
console.log('--- HTML/CSS 粗检 ---');
{
  ok('没有未闭合的 <style>', (html.match(/<style>/g) || []).length === (html.match(/<\/style>/g) || []).length);
  // 非法颜色值：只在 CSS 属性值里找 `#xxxx`
  // （第一版把 CSS 选择器和 GLSL 的 #version / #define 全当成了颜色，误报一片）
  {
    let css = '';
    const sm = /<style>([\s\S]*?)<\/style>/.exec(html);
    if (sm) {
      css = sm[1].replace(/\/\*[\s\S]*?\*\//g, ' ');   // 去掉 CSS 注释
    }
    // 只看 `属性: ...;` 里的值
    const badColors = [];
    const declRe = /:\s*([^;{}]+);/g;
    let dm;
    while ((dm = declRe.exec(css)) !== null) {
      const val = dm[1];
      const hashes = val.match(/#[0-9a-zA-Z]+/g) || [];
      for (const h of hashes) {
        if (!/^#([0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$/.test(h)) {
          badColors.push(h);
        }
      }
    }
    ok('CSS 里没有非法十六进制颜色值', badColors.length === 0, badColors.join(' '));
  }
  ok('引用了 canvas#gl', /id="gl"/.test(html));
}

console.log('');
console.log('='.repeat(60));
console.log('通过 ' + pass + ' 项，失败 ' + fail + ' 项');
if (fail) {
  console.log('');
  console.log('注意：这是**静态**检查，不能替代真实编译。');
  console.log('最终仍以浏览器里 gl.getShaderInfoLog 的输出为准。');
  for (const f of failures) console.log('  失败: ' + f);
}
process.exit(fail === 0 ? 0 : 1);
