// @ts-check
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import expressiveCode from 'astro-expressive-code';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';
import { remarkAlert } from 'remark-github-blockquote-alert';
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
  integrations: [
    // Code frames with titles, copy buttons, and terminal styling, themed to the brand.
    expressiveCode({
      themes: ['github-dark-default'],
      useThemedScrollbars: false,
      styleOverrides: {
        borderRadius: '0.75rem',
        borderColor: 'rgba(255,255,255,0.08)',
        codeBackground: '#0c0d10',
        codeFontFamily: "'JetBrains Mono Variable', 'JetBrains Mono', ui-monospace, monospace",
        codeFontSize: '0.8125rem',
        codeLineHeight: '1.7',
        uiFontFamily: "'Inter Variable', Inter, ui-sans-serif, system-ui, sans-serif",
        focusBorder: '#2dd4bf',
        frames: {
          shadowColor: 'transparent',
          editorTabBarBackground: '#101115',
          editorActiveTabBackground: '#0c0d10',
          editorActiveTabIndicatorTopColor: '#2dd4bf',
          terminalTitlebarBackground: '#101115',
          terminalBackground: '#0c0d10',
          terminalTitlebarBorderBottomColor: 'rgba(255,255,255,0.06)',
          inlineButtonBorder: 'rgba(255,255,255,0.15)',
          tooltipSuccessBackground: '#0d9488',
        },
      },
    }),
    sitemap(),
  ],
  markdown: {
    processor: unified({
      remarkPlugins: [
        [remarkDocLinks, { base, docsRoot: fileURLToPath(new URL('../docs', import.meta.url)) }],
        // GitHub-style callouts (> [!NOTE]) render the same here and on GitHub.
        remarkAlert,
      ],
      rehypePlugins: [
        [rehypeAutolinkHeadings, { behavior: 'append', properties: { className: ['heading-anchor'], ariaHidden: 'true', tabIndex: -1 }, content: { type: 'text', value: '#' } }],
      ],
    }),
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
