/**
 * The first paragraph after a doc's title, as plain text. Each doc opens
 * with a one-paragraph summary; it becomes the page's meta description and
 * its card text on the docs home.
 */
export function summaryOf(markdown: string): string {
  const lines = markdown.split('\n');
  const start = lines.findIndex((l) => l.startsWith('# '));
  const paragraph: string[] = [];
  for (const line of lines.slice(start + 1)) {
    const text = line.trim();
    if (!text) {
      if (paragraph.length) break;
      continue;
    }
    if (/^(#|```|\||>|[-*] |\d+\. )/.test(text)) {
      if (paragraph.length) break;
      continue;
    }
    paragraph.push(text);
  }
  return paragraph
    .join(' ')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[`*_]/g, '')
    .trim();
}
