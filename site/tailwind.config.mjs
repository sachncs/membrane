/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,ts,tsx,vue,svelte}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Graphite neutrals.
        ink: {
          950: '#08090b',
          900: '#0c0d10',
          850: '#101115',
          800: '#14161a',
          750: '#191b20',
          700: '#202329',
          600: '#2b2f36',
          500: '#40454e',
          400: '#646a75',
          300: '#8d939e',
          200: '#b9bec7',
          100: '#e3e6eb',
          50: '#f5f6f8',
        },
        // The one brand accent: membrane teal.
        brand: {
          50: '#effdfa',
          100: '#c9f9ef',
          200: '#94f1e0',
          300: '#5be3cd',
          400: '#2dd4bf',
          500: '#14b8a6',
          600: '#0d9488',
          700: '#0f766e',
          800: '#115e59',
          900: '#134e4a',
        },
        // Secondary signal colours, used sparingly in diagrams.
        signal: {
          amber: '#f5b454',
          rose: '#f47c8a',
          sky: '#7cc4fa',
        },
      },
      fontFamily: {
        sans: ['Inter Variable', 'Inter', '-apple-system', 'BlinkMacSystemFont', 'Helvetica Neue', 'Arial', 'sans-serif'],
        display: ['Inter Variable', 'Inter', '-apple-system', 'BlinkMacSystemFont', 'Helvetica Neue', 'Arial', 'sans-serif'],
        mono: ['JetBrains Mono Variable', 'JetBrains Mono', 'SF Mono', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem', letterSpacing: '0.02em' }],
        'display-xl': ['clamp(2.5rem, 5.2vw, 4.25rem)', { lineHeight: '1.04', letterSpacing: '-0.035em' }],
        'display-lg': ['clamp(2rem, 3.6vw, 3rem)', { lineHeight: '1.08', letterSpacing: '-0.03em' }],
        'display-md': ['clamp(1.625rem, 2.6vw, 2.25rem)', { lineHeight: '1.12', letterSpacing: '-0.025em' }],
        'display-sm': ['clamp(1.25rem, 1.8vw, 1.5rem)', { lineHeight: '1.25', letterSpacing: '-0.015em' }],
      },
      letterSpacing: {
        tightest: '-0.04em',
        tighter: '-0.025em',
      },
      borderRadius: {
        '4xl': '2rem',
      },
      boxShadow: {
        card: '0 1px 0 0 rgba(255,255,255,0.04) inset, 0 20px 40px -24px rgba(0,0,0,0.6)',
        window: '0 1px 0 0 rgba(255,255,255,0.06) inset, 0 40px 80px -40px rgba(0,0,0,0.8), 0 0 0 1px rgba(255,255,255,0.06)',
        'brand-ring': '0 0 0 1px rgba(45,212,191,0.35), 0 8px 30px -12px rgba(45,212,191,0.45)',
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
};
