import { absoluteSiteUrl } from "../lib/site";
export function GET({site}:{site?:URL}) {
  return new Response(`User-agent: *
Allow: /

Sitemap: ${absoluteSiteUrl("/sitemap.xml",site)}
`, {headers:{"Content-Type":"text/plain; charset=utf-8"}});
}
