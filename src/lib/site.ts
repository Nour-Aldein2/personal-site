export const siteSettings = {
  siteTitle: "Noor Aldeen Almusleh",
  siteTagline: "AI/ML Engineer and Researcher",
  siteDescription:
    "AI/ML engineer and researcher Noor Aldeen Almusleh works across reinforcement learning, computer vision, statistical modelling, MLOps, and computer graphics.",
  authorName: "Noor Aldeen Almusleh",
  authorRole: "AI/ML Engineer and Researcher",
  authorImage: "/images/profile_pic.jpg",
  cvFilePath: "/Noor_CV.pdf",
  contactHref: "mailto:contact@nourd.uk",
  location: "Bath, UK",
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
    "AI/ML engineer",
    "machine learning engineer",
    "MSc Artificial Intelligence",
    "MSc AI",
    "University of Bath",
    "reinforcement learning",
    "computer vision",
    "natural language processing",
    "NLP",
    "MLOps",
    "statistical modelling",
    "statistical modeling",
    "control and optimisation",
    "control and optimization",
    "computer graphics",
    "scientific computing",
    "generative modelling",
    "generative modeling",
    "chemical recipe optimisation",
    "chemical recipe optimization",
    "MXene",
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
