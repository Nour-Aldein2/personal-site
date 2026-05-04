import type { CollectionEntry } from "astro:content";

export type ArticleEntry = CollectionEntry<"articles">;

export function sortArticles(entries: ArticleEntry[]) {
  return [...entries].sort(
    (a, b) =>
      new Date(b.data.publishDate).getTime() - new Date(a.data.publishDate).getTime(),
  );
}

export function formatDate(date: Date) {
  return new Intl.DateTimeFormat("en", {
    year: "numeric",
    month: "long",
    day: "numeric",
  }).format(date);
}

export function getReadingTime(body: string) {
  const clean = body.replace(/[`#*_>[\]()!-]/g, " ");
  const words = clean.split(/\s+/).filter(Boolean).length;
  const minutes = Math.max(1, Math.round(words / 220));
  return `${minutes} min read`;
}

export function getRelatedArticles(current: ArticleEntry, allEntries: ArticleEntry[]) {
  const currentTags = new Set(current.data.tags);

  return sortArticles(
    allEntries
      .filter((entry) => entry.data.slug !== current.data.slug)
      .map((entry) => ({
        entry,
        score: entry.data.tags.reduce(
          (total, tag) => total + (currentTags.has(tag) ? 1 : 0),
          0,
        ),
      }))
      .filter(({ score }) => score > 0)
      .sort((a, b) => b.score - a.score)
      .map(({ entry }) => entry),
  ).slice(0, 2);
}

export function getPrevNextArticles(current: ArticleEntry, entries: ArticleEntry[]) {
  const sorted = sortArticles(entries);
  const index = sorted.findIndex((entry) => entry.data.slug === current.data.slug);

  return {
    newer: index > 0 ? sorted[index - 1] : undefined,
    older: index >= 0 && index < sorted.length - 1 ? sorted[index + 1] : undefined,
  };
}
