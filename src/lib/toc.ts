import type { MarkdownHeading } from "astro";

export interface TocItem extends MarkdownHeading {
  children: TocItem[];
}

/**
 * Converts Astro's flat heading array into a nested table of contents.
 *
 * ##   -> top-level section
 * ###  -> subsection
 * #### -> nested subsection
 */
export function buildToc(
  headings: readonly MarkdownHeading[] = [],
): TocItem[] {
  const roots: TocItem[] = [];
  const stack: TocItem[] = [];

  for (const heading of headings) {
    // Don't include the article's H1/title in the TOC.
    if (heading.depth === 1) {
      stack.length = 0;
      continue;
    }

    if (heading.depth < 2 || heading.depth > 6) {
      continue;
    }

    const item: TocItem = {
      ...heading,
      children: [],
    };

    // Walk back up the tree until we find the parent heading.
    while (
      stack.length > 0 &&
      stack[stack.length - 1].depth >= item.depth
    ) {
      stack.pop();
    }

    const parent = stack[stack.length - 1];

    if (parent) {
      parent.children.push(item);
    } else {
      roots.push(item);
    }

    stack.push(item);
  }

  return roots;
}