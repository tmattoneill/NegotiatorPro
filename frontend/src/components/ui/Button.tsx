import type { ButtonHTMLAttributes, PropsWithChildren } from 'react'
import { cn } from './cn'

type Variant = 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger' | 'success'
type Size = 'sm' | 'md' | 'lg' | 'icon'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  size?: Size
}

export default function Button({
  className,
  variant = 'primary',
  size = 'md',
  children,
  ...props
}: PropsWithChildren<ButtonProps>) {
  const base = 'inline-flex items-center justify-center rounded-md font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-60 disabled:cursor-not-allowed'

  const variants: Record<Variant, string> = {
    primary: 'bg-chat-primary text-white hover:bg-chat-primary-hover focus:ring-chat-primary',
    secondary: 'bg-chat-muted text-chat-foreground hover:bg-chat-muted focus:ring-chat-border',
    outline: 'border border-chat-border bg-transparent text-chat-foreground hover:bg-chat-muted focus:ring-chat-border',
    ghost: 'bg-transparent text-chat-foreground hover:bg-chat-muted focus:ring-chat-border',
    danger: 'bg-danger text-danger-foreground hover:opacity-90 focus:ring-danger',
    success: 'bg-success text-success-foreground hover:opacity-90 focus:ring-success',
  }

  const sizes: Record<Size, string> = {
    sm: 'h-9 px-3 text-sm',
    md: 'h-10 px-4 text-sm',
    lg: 'h-11 px-5 text-base',
    icon: 'h-10 w-10',
  }

  return (
    <button className={cn(base, variants[variant], sizes[size], className)} {...props}>
      {children}
    </button>
  )
}

