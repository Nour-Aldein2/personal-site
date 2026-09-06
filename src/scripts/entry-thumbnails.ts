/** Images are optional. Never leave a broken-image box in the reading flow. */
function initialiseEntryThumbnails() {
  document.querySelectorAll<HTMLImageElement>("[data-entry-thumbnail]").forEach(image => {
    if (image.dataset.fallbackReady) return;
    image.dataset.fallbackReady = "true";
    const useTextOnly = () => image.closest("[data-entry-row]")?.classList.add("entry-row--text-only");
    if (image.complete && image.naturalWidth === 0) useTextOnly();
    else image.addEventListener("error", useTextOnly, { once: true });
  });
}
initialiseEntryThumbnails();
document.addEventListener("astro:page-load", initialiseEntryThumbnails);
