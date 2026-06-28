export interface ProjectEntry {
  title: string;
  description: string;
  tags: string[];
  status: string;
  href?: string;
}

export const featuredProjects: ProjectEntry[] = [
  {
    title: "Optimising MXene Stability with Deep Q-Learning",
    description:
      "A deep Q-learning agent trained in a custom environment to optimize MAX-to-MXene synthesis and enhance stability, leveraging a novel, custom-compiled dataset from the literature. The project will end with experimental validation of the agent's performance.",
    tags: ["Optimisation", "Deep RL", "Chemistry"],
    status: "Research Project",
  },
  {
    title: "Automating Repetitive Line Drawing from a Reference and a Character Design",
    description:
      "A PPO agent trained in a custom simulated environment to imitate in-between artists and draw line art from a reference and a character design.",
    tags: ["Computer Graphics", "Deep RL", "PPO"],
    status: "Research Project",
  },
  {
    title: "Image Fusion for Cloud Removal from Satellite Images",
    description:
      "Benchmarking AI techniques from existing literature to compare their baseline results. The project also is focusing on testing and adapting novel image fusion techniques derived from other fields to evaluate their direct application in satellite image processing.",
    tags: ["Computer Vision", "AI", "Data Fusion"],
    status: "Project",
  },
  {
    title: "Quantum Machine Learning with Qiskit",
    description:
      "Hands-on quantum machine learning work using Qiskit, covering classifiers, optimization methods, and applied experimentation with quantum circuits.",
    tags: ["Qiskit", "Quantum ML", "Quantum Computing"],
    status: "Project",
  },
];
