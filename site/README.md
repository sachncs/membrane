# Membrane

Global Contextual Memory Fabric — premium marketing site.

## Stack

- [Astro](https://astro.build) for static site generation
- [Tailwind CSS](https://tailwindcss.com) for styling
- TypeScript for maintainability
- Self-hosted Inter & JetBrains Mono variable fonts

## Develop

```bash
cd site
npm install
npm run dev
```

The dev server runs on http://localhost:4321.

## Build

```bash
cd site
npm run build
```

Static output is emitted to `site/dist/` and is uploaded by the GitHub
Pages workflow as the published site.

## Structure

```
site/
├── public/                # static assets (favicon, og, fonts)
├── src/
│   ├── components/        # page sections (Hero, Features, ...)
│   ├── layouts/           # base layout
│   ├── pages/             # routes (index.astro)
│   └── styles/            # global CSS
├── astro.config.mjs
└── tailwind.config.mjs
```
