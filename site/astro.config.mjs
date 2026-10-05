// @ts-check
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import { unified } from '@astrojs/markdown-remark';
import remarkDocLinks from './src/lib/remark-doc-links.mjs';

const base = '/membrane';

// The config file is not bundled, so it can read the repository's
// pyproject.toml reliably; the version is injected at build time.
const pyproject = readFileSync(new URL('../pyproject.toml', import.meta.url), 'utf8');
const version = pyproject.match(/^version\s*=\s*"([^"]+)"/m)?.[1] ?? 'dev';

export default defineConfig({
  site: 'https://sachncs.github.io',
  base,
  output: 'static',
  build: {
    format: 'directory',
  },
  trailingSlash: 'ignore',
  integrations: [sitemap()],
  markdown: {
    processor: unified({
      remarkPlugins: [[remarkDocLinks, { base, docsRoot: fileURLToPath(new URL('../docs', import.meta.url)) }]],
    }),
    shikiConfig: { theme: 'github-dark-dimmed', wrap: true },
  },
  vite: {
    define: {
      'import.meta.env.MEMBRANE_VERSION': JSON.stringify(version),
    },
    css: {
      postcss: './postcss.config.cjs',
    },
  },
});
