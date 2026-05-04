import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const articles = defineCollection({
  loader: glob({
    pattern: "**/*.md",
    base: "./src/content/articles",
  }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    slug: z.string(),
    publishDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    templateType: z.enum(["case-study", "tutorial", "research-note"]),
    tags: z.array(z.string()),
    featured: z.boolean().optional().default(false),
    draft: z.boolean().optional().default(false),
    heroImage: z.string().optional(),
  }),
});

export const collections = { articles };
