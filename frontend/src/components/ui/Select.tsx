import type { SelectHTMLAttributes } from 'react'
import { cn } from './cn'

export default function Select({ className, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={cn(
        'block w-full rounded-md border border-chat-border bg-white text-chat-foreground',
        'px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-chat-primary focus:border-chat-primary',
        'disabled:opacity-60 disabled:cursor-not-allowed',
        className,
      )}
      {...props}
    />
  )
}

