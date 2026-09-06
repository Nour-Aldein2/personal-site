/* Load once after the shared header, or with <script defer src="...">.
   Also handles Astro client-side page swaps, back/forward restoration and resize.
   Safe to initialise again: the previous listener/observer set is removed first. */
(() => {
  const key = "__personalSiteScrollHeaderCleanup";
  if (typeof window[key] === "function") window[key]();
  const root = document.documentElement;
  const threshold = 8; // Ignore tiny scroll offsets and mobile overscroll at the top.
  let header = null;
  let observer = null;
  let frame = 0;

  const update = () => {
    frame = 0;
    if (!header) return;
    const state = window.scrollY > threshold ? "true" : "false";
    if (header.dataset.scrolled !== state) header.dataset.scrolled = state;
  };
  const schedule = () => {
    if (!frame) frame = window.requestAnimationFrame(update);
  };
  const measure = () => {
    if (!header) return;
    // Keep in-page headings and the article TOC clear of the responsive header.
    const height = Math.ceil(header.getBoundingClientRect().height);
    root.style.setProperty("--sticky-header-height", `${height}px`);
  };
  const mount = () => {
    observer?.disconnect();
    header = document.querySelector(".site-header");
    if (!header) return;
    update();
    header.dataset.scrollHeader = "ready";
    measure();
    if ("ResizeObserver" in window) {
      observer = new ResizeObserver(measure);
      observer.observe(header);
    }
    schedule();
  };
  const onResize = () => { measure(); schedule(); };
  window.addEventListener("scroll", schedule, { passive: true });
  window.addEventListener("resize", onResize, { passive: true });
  window.addEventListener("pageshow", mount);
  document.addEventListener("astro:page-load", mount);
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount, { once: true });
  } else {
    mount();
  }
  window[key] = () => {
    window.removeEventListener("scroll", schedule);
    window.removeEventListener("resize", onResize);
    window.removeEventListener("pageshow", mount);
    document.removeEventListener("astro:page-load", mount);
    document.removeEventListener("DOMContentLoaded", mount);
    observer?.disconnect();
    if (frame) window.cancelAnimationFrame(frame);
  };
})();
