export const siteSettings = {
  siteTitle: "Noor Aldeen Almusleh",
  siteTagline: "AI Researcher and AI Engineer",
  siteDescription:
    "Noor Aldeen Almusleh is an AI researcher and AI engineer focused on machine learning, NLP, explainability, astrophysics-informed research, and technical writing.",
  authorName: "Noor Aldeen Almusleh",
  authorRole: "AI Researcher and AI Engineer",
  authorImage: "/images/profile_pic.png",
  cvFilePath: "/Noor_CV.pdf",
  location: "Dubai, UAE",
  sameAs: [
    "https://github.com/Nour-Aldein2",
    "https://www.linkedin.com/in/nour-aldein-b15361123/",
  ],
  alternateNames: [
    "Nour Aldein Almusleh",
    "Noor Almusleh",
    "Nour Aldeen",
    "Nour Aldein",
    "Noor Aldein",
    "Nour Aldein Almusleh",
  ],
  keywords: [
    "Noor Aldeen Almusleh",
    "Noor Almusleh",
    "Nour Aldeen",
    "Nour Aldein",
    "Noor Aldein",
    "Nour Aldein Almusleh",
    "AI researcher",
    "AI engineer",
    "machine learning engineer",
    "NLP",
    "technical writing",
  ],
};

function ensureTrailingSlash(value: string) {
  return value.endsWith("/") ? value : `${value}/`;
}

function stripLeadingSlash(value: string) {
  return value.replace(/^\/+/, "");
}

export function sitePath(path = "/") {
  const base = import.meta.env.BASE_URL || "/";
  const cleanBase = ensureTrailingSlash(base);
  const cleanPath = stripLeadingSlash(path);

  return cleanPath ? `${cleanBase}${cleanPath}` : cleanBase;
}

export function absoluteSiteUrl(path = "/", site?: URL | string) {
  const siteRoot = site
    ? new URL(ensureTrailingSlash(site instanceof URL ? site.toString() : site))
    : new URL("https://your-domain.example/");
  const cleanPath = stripLeadingSlash(path);

  return cleanPath ? new URL(cleanPath, siteRoot).toString() : siteRoot.toString();
}

export const articleTemplateLabels = {
  "case-study": "Case Study",
  tutorial: "Tutorial",
  "research-note": "Research Note",
} as const;
