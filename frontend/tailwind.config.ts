import type { Config } from 'tailwindcss'

export default {
  darkMode: 'class',
  content: [
    './index.html',
    './src/**/*.{ts,tsx,js,jsx}'
  ],
  theme: {
    extend: {
      colors: {
        // Design tokens mapped to CSS variables
        'chat-primary': 'hsl(var(--chat-primary))',
        'chat-primary-hover': 'hsl(var(--chat-primary-hover))',
        'chat-bg': 'hsl(var(--chat-bg))',
        'chat-foreground': 'hsl(var(--chat-foreground))',
        'chat-muted': 'hsl(var(--chat-muted))',
        'chat-muted-foreground': 'hsl(var(--chat-muted-foreground))',
        'chat-border': 'hsl(var(--chat-border))',
        'chat-sidebar': 'hsl(var(--chat-sidebar))',
        'chat-card': 'hsl(var(--chat-card))',
        // Messages
        'message-user-bg': 'hsl(var(--message-user-bg))',
        'message-user-text': 'hsl(var(--message-user-text))',
        'message-assistant-bg': 'hsl(var(--message-assistant-bg))',
        'message-assistant-text': 'hsl(var(--message-assistant-text))',
        // Code blocks
        'code-bg': 'hsl(var(--code-bg))',
        'code-text': 'hsl(var(--code-text))',
        'code-border': 'hsl(var(--code-border))',
        'code-header': 'hsl(var(--code-header))',
        // Status
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
        sm: {
          css: {
            // Constrain heading sizes for chat bubbles (prose-sm base = 14px)
            h1: { fontSize: '1.2em',   fontWeight: '700', marginTop: '1em',    marginBottom: '0.4em' },
            h2: { fontSize: '1.125em', fontWeight: '600', marginTop: '1em',    marginBottom: '0.35em' },
            h3: { fontSize: '1.05em',  fontWeight: '600', marginTop: '0.85em', marginBottom: '0.3em' },
            h4: { fontSize: '1em',     fontWeight: '600', marginTop: '0.75em', marginBottom: '0.25em' },
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

