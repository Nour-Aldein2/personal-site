# Noor Aldeen — seamless brown/grid website release

Cumulative website source for `Nour-Aldein2/personal-site`. This replaces the earlier
teal/mint/lavender experiments with the final approved design. No individual patches
need to be applied on top of this release.

## Included design

Light mode uses white, **#7C2710** accents, a soft warm page wash, and a faint brown
grid. Dark mode retains the approved warm background, **#CC9075** accents and grid,
and its restrained matte colour wash. The header and page share one continuous
background at the top. After 8px of scrolling, the sticky navigation gains an opaque
theme-matched surface and a fine divider. Returning to the top removes them without
changing the header's height. The light-mode scrolled header is white, not green.

The same palette, canvas and header controller are imported by `BaseLayout.astro`:
Home, Articles, every article detail page, Publications, the CV redirect page, and
404 all share them. Legacy `.html` redirect fallbacks receive the same palette and
canvas at build time. Original photographs, article figures, code syntax colours,
and the CV PDF are not recoloured.

The compact “Noor Aldeen” name beside the role, 4pt reduction to the introduction's
full name, unboxed introduction, borderless article/project/publication lists,
removed type labels, and small grey dates are retained. The removed writing-section
sentence stays removed. Grids are static, weaker on mobile, and hidden in print and
high-contrast modes. Reduced-motion preferences disable header transitions.

## Important: source package, not a prebuilt `dist/`

All website page sources, shared components/styles/scripts, article content,
configuration, tests and GitHub Pages workflow are included. **Original images,
CV PDF and the upstream dependency lockfile are not embedded in this ZIP.**
They remain in your existing repository; the installer preserves them. When they
are missing, setup restores the originals from an immutable repository revision,
not a moving branch. Internet access is needed for that restoration and npm packages.

There are no preview placeholder photos, reconstructed figures, font files,
`node_modules`, or prebuilt `dist/` files in this release. The unrelated `dashboard/`
project is not included or modified. Source and browser-fixture tests passed, but
Astro compilation could not run in the packaging environment because downloads
were unavailable. Run the local build below before deployment. Details are in
`docs/VALIDATION.md`.

## Recommended: update your existing Git checkout

Use **Node.js 22 (22.12 or newer)**. `.nvmrc` selects the Node 22 line.

Extract the ZIP outside your existing repository. In the extracted `personal-site`
folder, run:

```sh
node apply-to-repository.mjs "/path/to/your/personal-site"
```

To inspect the planned file changes without writing anything:

```sh
node apply-to-repository.mjs "/path/to/your/personal-site" --dry-run
```

The installer backs up overwritten files in `.site-update-backups/`, preserves
original media, your lockfile, `.git` history, `.env`, and unrelated files, and does
not commit or push. Save or commit any unrelated local work first. New article files
not present in this package are left intact; review existing content before merging.

Then open a terminal in **your existing repository** and run:

```sh
npm run setup
npm ci
npm run check
npm run preview
```

Open the local address printed in the terminal, normally `http://localhost:4321/`.
Check both themes, scrolling down and back up, mobile widths, Publications, Articles,
an article, and the CV link. Press Ctrl+C to stop preview. Use `npm run dev` for
live updates while editing. Run `npm run check` again after editing before previewing
the new production output.

If setup reports an incompatible lockfile, do not delete it blindly. This release
pins Astro **6.2.1**, matching the referenced original lockfile. The setup script
normalises only that lockfile's root dependency label and preserves transitive
versions and integrity hashes. It refuses a lockfile resolving another Astro version.

After the build succeeds and the preview looks right:

```sh
git status
git diff
# Stage only this site's reviewed changes; keep unrelated work out of the commit.
git add .
git commit -m "Apply seamless header and consistent brown themes across the site"
git push origin main
```

Manual copying also works: merge the extracted folder's contents into the repository
root, including `.github`. Do not nest the whole extracted folder in the repository,
replace `.git`, or delete the original `public/` directory. The installer is safer
because it makes backups. Check for other deployment workflows left in your checkout
and keep only the intended Pages workflow active.

## GitHub Pages

In **Settings → Pages → Build and deployment**, select **GitHub Actions**.
The workflow defaults to the existing custom domain:

```text
SITE_URL=https://nourd.uk
BASE_PATH=/
```

These are overridable repository Actions variables. Keep your existing custom-domain
and DNS configuration. The archive cannot change GitHub settings or DNS records.

A push to `main` triggers setup, `npm ci`, tests, the Astro build, and generated-page/
local-link checks before deployment. Pull requests run checks but are not deployed.
If any step fails, deployment does not run. The workflow can also be started manually.
It restores missing originals automatically, so no manual image substitutions are
needed when you merge this package into a checkout that already contains the media.

For a project-pages address instead of a custom domain:

```text
SITE_URL=https://Nour-Aldein2.github.io
BASE_PATH=/personal-site/
```

Links, asset paths, sitemap, canonical URLs and legacy redirects respect the base
path. The output checker removes `dist/CNAME` for a `github.io` deployment.

## Run the extracted package without an existing checkout

From the extracted `personal-site` folder:

```sh
node prepare-site.mjs
npm run preview
```

The helper restores originals, installs exact dependencies, tests and builds. It
never pushes or deploys. To copy missing originals from a local checkout instead:

```sh
node prepare-site.mjs --from "/path/to/original/personal-site"
```

After a successful build the contents of `dist/` can be uploaded to a static host.
Do not upload `src/` directly as a static site. This release is configured for GitHub
Actions to build from source, not for serving the repository root as plain HTML.

## Shared source map

| File | Responsibility |
| --- | --- |
| `src/styles/theme.css` | Light/dark palette and typography tokens |
| `src/styles/global.css` | Shared page/content typography and layout |
| `src/styles/header.css` | Compact navigation and dark matte wash |
| `src/styles/canvas.css` | Continuous, softly fading background and both grids |
| `src/styles/scroll-header.css` | Seamless and scrolled header states |
| `src/scripts/scroll-header.js` | Scroll state, resize measurement and cleanup |
| `src/styles/entries.css` | Borderless article and project rows |
| `src/layouts/BaseLayout.astro` | Shared page frame and before-paint theme setup |
| `scripts/` | Asset setup and generated-output validation |
| `.github/workflows/deploy.yml` | PR validation and GitHub Pages deployment |

Edit shared tokens rather than adding page-specific palette overrides. Future pages
using `BaseLayout` (or `ArticleLayout`, which uses it) inherit the design automatically.
