export const REPO_URL = 'https://github.com/sachncs/membrane';
export const BLOB_URL = `${REPO_URL}/blob/master`;
export const TREE_URL = `${REPO_URL}/tree/master`;

/** Version from pyproject.toml, injected by astro.config.mjs at build time. */
export const VERSION: string = import.meta.env.MEMBRANE_VERSION ?? 'dev';

/** Prefix an internal path with the deployment base (e.g. /membrane). */
export function withBase(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  return `${base}${path.startsWith('/') ? path : `/${path}`}`;
}

export const INSTALL_COMMANDS = `git clone ${REPO_URL}.git
cd membrane
uv sync --frozen --extra server   # Python 3.14 + locked deps
source .venv/bin/activate`;
