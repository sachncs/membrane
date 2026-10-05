import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export const REPO_URL = 'https://github.com/sachncs/membrane';
export const BLOB_URL = `${REPO_URL}/blob/master`;
export const TREE_URL = `${REPO_URL}/tree/master`;

/** Version from pyproject.toml, so the site never advertises a stale release. */
export const VERSION = (() => {
  const pyproject = readFileSync(fileURLToPath(new URL('../../../pyproject.toml', import.meta.url)), 'utf8');
  const match = pyproject.match(/^version\s*=\s*"([^"]+)"/m);
  return match ? match[1] : 'dev';
})();

/** Prefix an internal path with the deployment base (e.g. /membrane). */
export function withBase(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  return `${base}${path.startsWith('/') ? path : `/${path}`}`;
}

export const INSTALL_COMMANDS = `git clone ${REPO_URL}.git
cd membrane
python -m venv .venv && source .venv/bin/activate
pip install -e ".[server]"`;
