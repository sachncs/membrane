// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://sachncs.github.io',
  base: '/membrane',
  output: 'static',
  build: {
    format: 'directory',
  },
  trailingSlash: 'ignore',
  integrations: [sitemap()],
  vite: {
    css: {
      postcss: './postcss.config.cjs',
    },
  },
});
