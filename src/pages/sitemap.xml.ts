import { getCollection } from "astro:content";
import { absoluteSiteUrl } from "../lib/site";

export async function GET({ site }: { site?: URL }) {
  const articles = await getCollection("articles", ({ data }) => !data.draft);

  const urls = [
    "/",
    "/publications/",
    "/contact/",
    "/articles/",
    ...articles.map((article) => `/articles/${article.data.slug}/`),
  ];

  const body = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls
  .map((path) => {
    const url = absoluteSiteUrl(path, site);
    return `  <url><loc>${url}</loc></url>`;
  })
  .join("\n")}
</urlset>`;

  return new Response(body, {
    headers: {
      "Content-Type": "application/xml; charset=utf-8",
    },
  });
}
