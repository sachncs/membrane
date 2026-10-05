// @ts-check
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import remarkDocLinks from './src/lib/remark-doc-links.mjs';

const base = '/membrane';

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
    remarkPlugins: [[remarkDocLinks, { base, docsRoot: fileURLToPath(new URL('../docs', import.meta.url)) }]],
    shikiConfig: { theme: 'github-dark-dimmed', wrap: true },
  },
  vite: {
    css: {
      postcss: './postcss.config.cjs',
    },
  },
});
