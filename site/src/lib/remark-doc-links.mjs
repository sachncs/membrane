// Rewrites relative links inside docs/*.md so the same files work both on
// GitHub and on the site:
//   other-doc.md#anchor     -> <base>/docs/other-doc/#anchor
//   ../docker-compose.yml   -> GitHub blob/tree URL
import path from 'node:path';
import { visit } from 'unist-util-visit';

const REPO = 'https://github.com/sachncs/membrane';

export default function remarkDocLinks({ base = '', docsRoot }) {
  const repoRoot = path.dirname(docsRoot);
  return (tree, file) => {
    const source = file.path || file.history?.[0];
    if (!source || !path.resolve(source).startsWith(docsRoot + path.sep)) return;
    visit(tree, ['link', 'definition'], (node) => {
      const url = node.url;
      if (!url || /^[a-z]+:/i.test(url) || url.startsWith('#') || url.startsWith('/')) return;
      const [target, hash = ''] = url.split('#');
      const abs = path.resolve(path.dirname(source), target);
      const anchor = hash ? `#${hash}` : '';
      if (abs.startsWith(docsRoot + path.sep) && abs.endsWith('.md')) {
        const slug = path.relative(docsRoot, abs).replace(/\.md$/, '').split(path.sep).join('/');
        node.url = `${base}/docs/${slug}/${anchor}`;
        return;
      }
      const rel = path.relative(repoRoot, abs).split(path.sep).join('/');
      const kind = target.endsWith('/') ? 'tree' : 'blob';
      node.url = `${REPO}/${kind}/master/${rel}${anchor}`;
    });
  };
}
