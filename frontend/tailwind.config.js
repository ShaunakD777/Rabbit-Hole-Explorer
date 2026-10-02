/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        canvas: 'var(--bg-canvas)',
        accent: {
          DEFAULT: '#6366f1', // indigo-500
          hover: '#4f46e5',
        },
        node: {
          default: '#64748b',   // slate-500
          inprogress: '#f59e0b', // amber-400
          learned: '#22c55e',   // green-500
          selected: '#6366f1',  // indigo-500
          trending: '#f97316',  // orange-500
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fadeIn 200ms ease-out',
        'slide-in': 'slideIn 220ms ease-out',
        'pulse-glow': 'pulseGlow 1.5s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: { from: { opacity: '0' }, to: { opacity: '1' } },
        slideIn: { from: { transform: 'translateX(100%)' }, to: { transform: 'translateX(0)' } },
        pulseGlow: {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(99,102,241,0.4)' },
          '50%': { boxShadow: '0 0 0 8px rgba(99,102,241,0)' },
        },
      },
    },
  },
  plugins: [],
}
