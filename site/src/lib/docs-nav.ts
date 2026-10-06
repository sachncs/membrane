/** Sidebar order for the docs. Every docs/*.md entry must appear here. */
export const DOCS_NAV: { title: string; items: { id: string; label: string }[] }[] = [
  {
    title: 'Get started',
    items: [
      { id: 'getting-started', label: 'Quickstart' },
      { id: 'faq', label: 'FAQ' },
    ],
  },
  {
    title: 'Guides',
    items: [
      { id: 'deployment', label: 'Deployment' },
      { id: 'security', label: 'Security & auth' },
      { id: 'consistency', label: 'Consistency levels' },
      { id: 'memory-api', label: 'Memory API & routing' },
      { id: 'disaggregation', label: 'Prefill / decode' },
      { id: 'plugins', label: 'Plugins' },
    ],
  },
  {
    title: 'Operations',
    items: [
      { id: 'operations/slo', label: 'SLOs' },
      { id: 'operations/capacity', label: 'Capacity planning' },
      { id: 'operations/backup-restore', label: 'Backup & restore' },
      { id: 'operations/upgrade', label: 'Upgrades' },
      { id: 'operations/incident-response', label: 'Incident response' },
    ],
  },
  {
    title: 'Reference',
    items: [
      { id: 'architecture', label: 'Architecture' },
      { id: 'wire-format', label: 'Wire format' },
      { id: 'api-stability', label: 'API stability' },
      { id: 'compat-matrix', label: 'Compatibility' },
      { id: 'release', label: 'Release process' },
    ],
  },
];

export const DOCS_ORDER = DOCS_NAV.flatMap((section) => section.items);
