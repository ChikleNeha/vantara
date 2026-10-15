/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['Sora', 'system-ui', 'sans-serif'],
        sans: ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      colors: {
        ink: { 950: '#150e2e', 900: '#1f1844', 700: '#4a4468', 500: '#756f92', 300: '#b9b5d0' },
        paper: { DEFAULT: '#f5f7fd', card: '#ffffff', line: '#e6e9f4' },
        pastel: { blue: '#6ea8ff', sky: '#eef5ff', peri: '#a3bdff', mist: '#f3f3f5' },
        risk: { high: '#c2432b', med: '#d39a1c', low: '#2e7d6a' },
        accent: { DEFAULT: '#4f8cff', soft: '#eef5ff', deep: '#1d4ed8' },
      },
      boxShadow: { card: '0 1px 0 rgba(21,14,46,0.03), 0 14px 32px -24px rgba(21,14,46,0.35)' },
    },
  },
  plugins: [],
}
