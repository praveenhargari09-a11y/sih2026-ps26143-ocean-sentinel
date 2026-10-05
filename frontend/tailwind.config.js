/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      colors: {
        // Ops base palette — deep near-black, not pure #000
        ops: {
          bg:      '#0d1117',   // main background
          panel:   '#161b22',   // panel surfaces
          raised:  '#1c2128',   // raised elements, hover targets
          border:  '#30363d',   // borders and dividers
          muted:   '#8b949e',   // secondary text
          subtle:  '#21262d',   // subtle backgrounds
        },
        // Primary accent — amber, reads well against dark ocean maps
        amber: {
          50:  '#fffbeb', 100: '#fef3c7', 200: '#fde68a',
          300: '#fcd34d', 400: '#fbbf24', 500: '#f59e0b',
          600: '#d97706', 700: '#b45309', 800: '#92400e', 900: '#78350f',
        },
        // Data accent — cyan for coordinates, MMSIs, timestamps
        cyan: {
          50:  '#ecfeff', 100: '#cffafe', 200: '#a5f3fc',
          300: '#67e8f9', 400: '#22d3ee', 500: '#06b6d4',
          600: '#0891b2', 700: '#0e7490', 800: '#155e75', 900: '#164e63',
        },
        // Danger — high-confidence suspect, errors
        danger: {
          DEFAULT: '#f85149',
          dim:     'rgba(248, 81, 73, 0.15)',
          border:  'rgba(248, 81, 73, 0.4)',
        },
        // Success / confirmed
        success: {
          DEFAULT: '#3fb950',
          dim:     'rgba(63, 185, 80, 0.15)',
        },
        // Drift colors
        drift: {
          hindcast: '#4b5563',  // muted gray — known past
          forecast: '#f59e0b',  // amber — uncertain future
        },
      },
      keyframes: {
        'spill-pulse': {
          '0%, 100%': { opacity: '1', strokeWidth: '2' },
          '50%':      { opacity: '0.4', strokeWidth: '3' },
        },
        'ping-slow': {
          '75%, 100%': { transform: 'scale(2)', opacity: '0' },
        },
        'step-glow': {
          '0%, 100%': { boxShadow: '0 0 4px rgba(251,191,36,0.4)' },
          '50%':      { boxShadow: '0 0 12px rgba(251,191,36,0.8)' },
        },
        'slide-in-right': {
          from: { transform: 'translateX(100%)' },
          to:   { transform: 'translateX(0)' },
        },
        'dropdown-in': {
          from: { opacity: '0', transform: 'translateY(-4px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'spill-pulse':     'spill-pulse 2.5s ease-in-out infinite',
        'ping-slow':       'ping-slow 2s cubic-bezier(0,0,0.2,1) infinite',
        'step-glow':       'step-glow 2s ease-in-out infinite',
        'slide-in-right':  'slide-in-right 0.2s ease-out',
        'dropdown-in':     'dropdown-in 0.15s ease-out',
      },
    },
  },
  plugins: [],
}
