/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['Fraunces', 'Georgia', 'serif'],
        sans: ['"Instrument Sans"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      colors: {
        ink: { 950: '#12161d', 900: '#1b212b', 700: '#3a4351', 500: '#6b7585', 300: '#aab2bf' },
        paper: { DEFAULT: '#f5f2ec', card: '#fffdf9', line: '#e4ded2' },
        risk: { high: '#c2432b', med: '#d39a1c', low: '#2e7d6a' },
        accent: { DEFAULT: '#2f5d8a', soft: '#dde8f2' },
      },
      boxShadow: { card: '0 1px 0 rgba(18,22,29,0.04), 0 8px 24px -16px rgba(18,22,29,0.25)' },
    },
  },
  plugins: [],
}
