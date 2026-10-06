# Membrane website

The source of <https://sachncs.github.io/membrane/>: the landing page and the documentation, which is rendered from the repository's [`docs/`](../docs/) directory (the single source of truth; there is no second copy).

## Stack

- [Astro](https://astro.build) static site, styled with [Tailwind CSS](https://tailwindcss.com)
- [Expressive Code](https://expressive-code.com) for code blocks (titles, copy buttons, brand theme)
- [Pagefind](https://pagefind.app) for search, indexed at build time
- `rehype-autolink-headings` for heading anchors, and `remark-github-blockquote-alert` for `> [!NOTE]` callouts that also render on GitHub
- Self-hosted Inter and JetBrains Mono variable fonts

## Develop

```bash
cd site
npm ci
npm run dev        # http://localhost:4321/membrane/
```

Search needs the Pagefind index, which only a full build creates:

```bash
npm run build      # astro build, then pagefind --site dist
npm run preview
npm run check      # type-check the site
```

The [Pages workflow](../.github/workflows/pages.yml) builds the site and deploys `dist/` on every push to `master` that touches `site/` or `docs/`.

## Structure

```text
site/
├── public/                  favicon set, logo, Open Graph image, web manifest
├── src/
│   ├── components/          landing sections, Nav, Logo, DocsHeader
│   ├── layouts/             BaseLayout, DocsLayout, DocsHome
│   ├── lib/
│   │   ├── docs-nav.ts      sidebar order; every docs/*.md page must be listed
│   │   ├── docs-summary.ts  a page's first paragraph, used as its description
│   │   └── remark-doc-links.mjs  rewrites docs links for the site
│   ├── pages/               index, docs home, docs/[...slug], 404
│   └── styles/global.css    design tokens, components, docs typography
├── astro.config.mjs
└── tailwind.config.mjs      brand palette (graphite and membrane teal)
```

## Adding a docs page

1. Add `docs/<name>.md`, starting with a `# Title` and a one-paragraph summary.
2. List it in `src/lib/docs-nav.ts`; the build fails if a page is missing.
3. Link to other pages with relative Markdown links (`security.md#authorization`); they work on GitHub and on the site.

## Icons

`public/favicon.svg` is the source mark, and `public/og.svg` is the source of the Open Graph image (`og.png`). The ICO (16 and 32 px), Apple touch, and manifest icons are rendered from the mark; `BaseLayout.astro` references them with a version query (`ICON_VERSION`), which you bump when the mark changes so browsers drop cached icons.
