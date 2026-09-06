# Release notes — 2026-09-06

## Final approved direction

The final white/brown light theme and warm brown dark theme replace all earlier
teal, mint and lavender header experiments. Background and grid span both header
and content. Only a scroll offset greater than 8px introduces the sticky bar's
opaque backing and fine divider. Its dimensions do not change between states.

## Changes from the previous deployment-source ZIP

- Replace the teal light tokens with #7C2710 and coordinated warm neutral tokens.
- Retain dark #CC9075 accents and approved warm gradient/texture treatment.
- Add shared `canvas.css` for static light/dark grids and two-sided/vertical fading.
- Add shared `scroll-header.css` and `scroll-header.js`, imported once in BaseLayout.
- Keep white light-mode header chrome after scroll; make both themes seamless at top.
- Remove the requested writing-section sentence from real homepage source.
- Apply the palette/canvas to legacy redirect fallbacks during postbuild.
- Preserve borderless lists, small metadata, removed labels and the 4pt name change.
- Add executable scroll-controller tests and require the current layout marker in
  generated HTML so stale pages do not silently pass output checks.
- Refresh setup/checkout Actions to the versions in their official documentation;
  retain the Pages upload/deploy versions documented by the Pages actions.

## Not changed

Article prose (other than the expressly removed homepage sentence), publication
records, project records, original asset paths, CV contents and contact links.
No repo push, account change, DNS change or live deployment was performed.
