# Original files and asset setup

Original repository: https://github.com/Nour-Aldein2/personal-site

Restoration revision: `7da6e6a09a84601fe8f569b3f5ce5798ea23d42b`

The source archive does **not** physically include the files listed below.
An existing repository checkout already contains them. The installer preserves
all existing copies, including customised images. When an original is absent,
`npm run setup` retrieves that file from the pinned revision over HTTPS, validates
its file signature and writes it atomically. A file that is already present but
invalid is reported, never silently replaced. Downloads are limited to 40 MB each.

## Original file manifest

- `package-lock.json`
- `public/Noor_CV.pdf`
- `public/images/profile_pic.jpg`
- `public/images/profile_pic.png`
- `public/images/profile_pic_old.png`
- `public/images/explain_BERT/tweet2.png`
- `public/images/explain_BERT/tweet3.png`
- `public/images/explain_BERT/tweet8.png`
- `public/images/explain_BERT/tweet9.png`
- `public/images/explain_BERT/tweets-class-distribution.png`
- `public/images/favicon_io_home/android-chrome-192x192.png`
- `public/images/favicon_io_home/android-chrome-512x512.png`
- `public/images/favicon_io_home/apple-touch-icon.png`
- `public/images/favicon_io_home/favicon-16x16.png`
- `public/images/favicon_io_home/favicon-32x32.png`
- `public/images/favicon_io_home/favicon.ico`

## Dependency reproducibility

The original lock resolves Astro **6.2.1**. This release changes the top-level
package declaration from `latest` to `6.2.1`. Setup makes the same root-label
change in the lockfile and preserves its exact transitive resolutions and
integrity hashes. A lock resolving another Astro version is rejected with an
explanation rather than overwritten.

Do not run `npm install` to replace the lock arbitrarily. Use `npm run setup`
followed by `npm ci`, as in the workflow. After first setup in a fresh repository,
commit the restored originals and normalised lockfile so later builds do not
need to restore them.

No website source code, credentials or private data are downloaded by setup.
No font files are requested by the asset restorer. The website references Google
Fonts at runtime and has local fallback fonts.

The published website uses local `/images/...` and `/Noor_CV.pdf` files, not
hotlinked production images and not the screenshot preview's placeholders.
