// Renders public/og.svg to public/og.png. Social networks do not render
// SVG Open Graph images, so the PNG is what BaseLayout advertises.
// Run after editing og.svg:  node scripts/render-og.mjs
import sharp from 'sharp';

await sharp('public/og.svg', { density: 144 }).resize(1200, 630).png().toFile('public/og.png');
console.log('wrote public/og.png');
