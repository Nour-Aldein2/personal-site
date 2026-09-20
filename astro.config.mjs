import { defineConfig } from "astro/config";
import { unified } from "@astrojs/markdown-remark";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

export default defineConfig({
  site: process.env.SITE_URL || "https://nourd.uk",
  base: process.env.BASE_PATH || "/",
  output: "static",
  compressHTML: true,
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex],
    }),
    shikiConfig: { themes: { light: "github-light", dark: "github-dark" } },
  },
  trailingSlash: "always",
});