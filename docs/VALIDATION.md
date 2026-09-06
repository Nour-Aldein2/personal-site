# Validation — seamless brown/grid source release

## Passed in the packaging environment

**51 Node tests**, using `npm test`. They cover shared imports, all content routes,
palette tokens, removal of obsolete theme colours and writing-section prose,
borderless entries, exact 4pt heading reduction, static grids/matte layers,
legacy redirects, generated-page/link validation, asset restoration safety,
lockfile normalisation and the backup-preserving repository installer.

Nine of those tests execute the **actual shipped scroll-header.js** in an event/DOM
harness: top/scrolled states, eight-pixel threshold, restored scroll position,
requestAnimationFrame batching, responsive height measurement, remount cleanup,
page swaps, DOMContentLoaded/ResizeObserver fallback and pages without a header.
The TAP output is included in `unit-tests.tap`.

**31 source files parsed for syntax** using TypeScript transpilation and PostCSS.
This checks TypeScript/JavaScript blocks and styles, not Astro template compilation,
module resolution or a production type check. See `syntax-checks.json`.

**76 browser-fixture checks** passed in Chromium: 6 page fixtures × 6 widths
(320, 390, 768, 1024, 1440 and 1920px) × 2 themes, plus four interaction/accessibility
checks. These load the actual shared CSS and JavaScript into HTML fixtures using
Playwright `set_content`. They check horizontal overflow, grid presence, palette,
transparent-at-top state, stable header height, sticky/solid scrolled state where
there is enough content to scroll, return-to-top state, theme controls, automatic
system theme, anchor offset, forced colours and print behaviour.

The theme-persistence fixture uses simulated storage because browser navigation/
network access is restricted here. These are **design-fixture checks, not screenshots
or end-to-end tests of an Astro-built website**. Fonts and media use the existing
fixture fallbacks, which are not included in production source. See `browser-checks.json`.

## Not verified here

A full Astro build, real npm installation, production media loading, live browser
navigation, GitHub Actions execution, DNS and deployment. The production build
command was attempted; its prebuild check correctly stopped because original media
and the upstream lockfile are absent from this source-only archive. Astro compilation
was therefore not executed. `build-attempt.log` records this explicitly. The environment
could not download the required original files or npm dependencies.

## Required final check on your computer or in CI

```sh
npm run setup
npm ci
npm run check
npm run preview
```

`check` runs tests plus the production build. The automatic postbuild step writes
the themed legacy redirects and validates actual generated pages, the current
shared-layout marker, required outputs, local link/image targets and base paths.
Deployment in the included workflow depends on that build job succeeding.

A passing fixture test must not be treated as a substitute for this production build.
