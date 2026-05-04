import { defineConfig } from "astro/config";

const rawSiteUrl = process.env.SITE_URL || "https://your-domain.example/";
const site = rawSiteUrl.endsWith("/") ? rawSiteUrl : `${rawSiteUrl}/`;
const derivedBasePath = new URL(site).pathname.replace(/\/$/, "") || "/";
const base = process.env.BASE_PATH || derivedBasePath;

export default defineConfig({
  site,
  base,
  output: "static",
});
