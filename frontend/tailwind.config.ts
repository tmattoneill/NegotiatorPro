import type { Config } from 'tailwindcss'

export default {
  darkMode: ['selector', '[data-theme="dark"]'],
  content: [
    './index.html',
    './src/**/*.{ts,tsx,js,jsx}'
  ],
  theme: {
    extend: {
      colors: {
        // Design tokens — OKLCH relative colour syntax preserves palette portability.
        // Swap palette.css and all colours update automatically.
        'chat-primary': 'oklch(from var(--color-accent) l c h / <alpha-value>)',
        'chat-primary-hover': 'oklch(from var(--color-accent-strong) l c h / <alpha-value>)',
        'chat-bg': 'oklch(from var(--color-background) l c h / <alpha-value>)',
        'chat-foreground': 'oklch(from var(--color-body) l c h / <alpha-value>)',
        'chat-muted': 'oklch(from var(--color-surface) l c h / <alpha-value>)',
        'chat-muted-foreground': 'oklch(from var(--color-muted) l c h / <alpha-value>)',
        'chat-border': 'oklch(from var(--color-border) l c h / <alpha-value>)',
        'chat-sidebar': 'oklch(from var(--color-heading) l c h / <alpha-value>)',
        'chat-card': 'oklch(from var(--color-surface) l c h / <alpha-value>)',
        // Strong (dark) end of the accent ramp — mid accent is too light for readable text.
        'message-user-bg': 'oklch(from var(--color-accent-strong) l c h / <alpha-value>)',
        'message-user-text': 'oklch(from var(--user-bubble-fg) l c h / <alpha-value>)',
        'message-assistant-bg': 'oklch(from var(--color-surface) l c h / <alpha-value>)',
        'message-assistant-text': 'oklch(from var(--color-body) l c h / <alpha-value>)',
        // Pop/highlight colour
        'pop': 'oklch(from var(--color-pop) l c h / <alpha-value>)',
        // Code blocks and status — kept on their own HSL vars (fixed, not palette-driven)
        'code-bg': 'hsl(var(--code-bg))',
        'code-text': 'hsl(var(--code-text))',
        'code-border': 'hsl(var(--code-border))',
        'code-header': 'hsl(var(--code-header))',
        success: 'hsl(var(--success))',
        'success-foreground': 'hsl(var(--success-foreground))',
        danger: 'hsl(var(--danger))',
        'danger-foreground': 'hsl(var(--danger-foreground))',
        warning: 'hsl(var(--warning))',
        'warning-foreground': 'hsl(var(--warning-foreground))',
      },
      borderRadius: {
        xl: '0.75rem',
      },
      boxShadow: {
        card: '0 1px 3px rgba(0,0,0,0.1)',
      },
      keyframes: {
        'fade-in': {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'fade-in': 'fade-in 0.2s ease-out',
      },
      typography: {
        DEFAULT: {
          css: {
            maxWidth: 'none',
            // Remove prose backtick pseudo-elements — we use a custom CodeBlock renderer
            'code::before': { content: 'none' },
            'code::after': { content: 'none' },
          },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
} satisfies Config
