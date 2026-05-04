import { absoluteSiteUrl } from "../lib/site";

export function GET({ site }: { site?: URL }) {
  const body = `User-agent: *
Allow: /

Sitemap: ${absoluteSiteUrl("/sitemap.xml", site)}
`;

  return new Response(body, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
    },
  });
}
