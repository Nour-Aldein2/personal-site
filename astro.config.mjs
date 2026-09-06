import { defineConfig } from "astro/config";
// Defaults preserve the existing custom domain. CI variables can override both.
export default defineConfig({
  site: process.env.SITE_URL || "https://nourd.uk",
  base: process.env.BASE_PATH || "/",
  output: "static",
  markdown: {
    shikiConfig: { themes: { light: "github-light", dark: "github-dark" } },
  },
  trailingSlash: "always",
});
