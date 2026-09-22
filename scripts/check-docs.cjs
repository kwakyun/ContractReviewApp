// Dependency-free checks for the entry documents maintained in this change.
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const files = ['README.md', 'AI_NOTES.md', 'CHANGELOG.md'];
const errors = [];
let links = 0;
for (const file of files) {
  const text = fs.readFileSync(path.join(root, file), 'utf8');
  let fence = null;
  let h1 = 0;
  for (const [index, line] of text.split(/\r?\n/).entries()) {
    const block = line.match(/^\s*(\x60{3,}|~{3,})/);
    if (block) { if (!fence) fence = block[1][0]; else if (fence === block[1][0]) fence = null; continue; }
    if (fence) continue;
    if (/^# /.test(line)) h1++;
    for (const match of line.matchAll(/!?\[([^\]]*)\]\((<[^>]+>|[^\s)]+)(?:\s+"[^"]*")?\)/g)) {
      let target = match[2].replace(/^<|>$/g, '');
      if (/^[a-z][a-z0-9+.-]*:|^#|^\/\//i.test(target)) continue;
      target = decodeURIComponent(target.split(/[?#]/)[0]);
      if (!target) continue;
      links++;
      const resolved = path.resolve(root, path.dirname(file), target);
      if (!fs.existsSync(resolved)) errors.push(file + ':' + (index+1) + ' missing: ' + target);
      if (match[0].startsWith('!') && !match[1].trim()) errors.push(file + ':' + (index+1) + ' image needs alt text');
    }
  }
  if (fence) errors.push(file + ': unclosed code fence');
  if (h1 !== 1) errors.push(file + ': expected one H1, got ' + h1);
}
if (errors.length) { console.error(errors.join('\n')); process.exit(1); }
console.log(files.length + ' entry documents checked; ' + links + ' local file links exist. External URLs, anchors and application behavior are not checked.');
