import type { HTMLAttributes, PropsWithChildren } from 'react'
import { cn } from './cn'

type Variant = 'default' | 'success' | 'danger' | 'warning' | 'muted' | 'outline'

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: Variant
}

export default function Badge({ variant = 'default', className, children, ...props }: PropsWithChildren<BadgeProps>) {
  const base = 'inline-flex items-center rounded px-2 py-0.5 text-xs font-medium'
  const variants: Record<Variant, string> = {
    default: 'bg-chat-muted text-chat-foreground',
    success: 'bg-success/10 text-success border border-success/30',
    danger: 'bg-danger/10 text-danger border border-danger/30',
    warning: 'bg-amber-500/10 text-amber-600 border border-amber-500/30',
    muted: 'bg-chat-muted text-chat-muted-foreground',
    outline: 'border border-chat-border text-chat-foreground',
  }
  return (
    <span className={cn(base, variants[variant], className)} {...props}>
      {children}
    </span>
  )
}

