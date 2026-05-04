export interface ProjectEntry {
  title: string;
  description: string;
  tags: string[];
  status: string;
  href?: string;
}

export const featuredProjects: ProjectEntry[] = [
  {
    title: "Lending Club Defaulters Prediction",
    description:
      "A machine learning workflow for credit-risk classification, combining exploratory analysis, feature preparation, and model evaluation for lending defaults.",
    tags: ["Machine Learning", "Risk", "scikit-learn"],
    status: "Project",
  },
  {
    title: "Paddy Disease Classification",
    description:
      "A computer vision project focused on image-based crop disease recognition using TensorFlow, data augmentation, and careful model evaluation.",
    tags: ["Computer Vision", "TensorFlow", "Agriculture"],
    status: "Project",
  },
  {
    title: "Quantum Machine Learning with Qiskit",
    description:
      "Hands-on quantum machine learning work using Qiskit, covering classifiers, optimization methods, and applied experimentation with quantum circuits.",
    tags: ["Qiskit", "Quantum ML", "Research"],
    status: "Research",
  },
];
