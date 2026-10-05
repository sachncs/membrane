import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';

// The repository's docs/ directory is the single source of truth: the
// site renders it directly instead of keeping a second copy.
const docs = defineCollection({
  loader: glob({ pattern: '**/*.md', base: '../docs' }),
});

export const collections = { docs };
