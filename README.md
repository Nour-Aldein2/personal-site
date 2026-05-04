# personal-site

Astro-based personal site, CV, and article portfolio for Noor Aldeen Almusleh.

## Structure

```text
.
├── .github/workflows/     # GitHub Pages deployment
├── public/                # Static files served as-is
├── src/
│   ├── components/        # Reusable UI pieces
│   ├── content/articles/  # Published article content
│   ├── layouts/           # Shared page layouts
│   ├── lib/               # Site data and helpers
│   ├── pages/             # Routes
│   └── styles/            # Global styling
├── article-templates/     # Private writing templates
├── dashboard/             # Separate non-site project, not deployed
├── astro.config.mjs
├── package.json
└── tsconfig.json
```

## Development

```bash
npm install
npm run dev
```

Then open `http://127.0.0.1:4321` or the local URL Astro prints.

## Build

```bash
SITE_URL="https://your-domain.example" npm run build
```

Set `SITE_URL` to your canonical production domain before deploying so canonical tags,
structured data, `robots.txt`, and the sitemap all point to the correct URL.

## GitHub Pages

Pushing to `main` now deploys the Astro site through GitHub Actions using
`.github/workflows/deploy.yml`.

Default behavior:

- If you do nothing, the workflow deploys to the standard project-pages URL:
  `https://<github-username>.github.io/personal-site/`
- If you later use a custom domain, set these repository variables in GitHub:
  - `SITE_URL`
  - `BASE_PATH`

Examples:

- Standard project site:
  - `SITE_URL=https://Nour-Aldein2.github.io/personal-site`
  - `BASE_PATH=/personal-site`
- Custom domain at root:
  - `SITE_URL=https://your-domain.example`
  - `BASE_PATH=/`

Important:

- Add your CV PDF at `public/Noor_CV.pdf` before pushing if you want the CV links to work.
- Contact links open the visitor's email client with `mailto:contact@nourd.uk`.

## Check Before Deploying

Run:

```bash
npm install
npm run dev
```

Then check these pages locally:

- `/`
- `/articles/`
- `/publications/`
- `/articles/bert-lime-disaster-tweets/`

Before pushing, also run:

```bash
npm run build
```
