import type { TextareaHTMLAttributes } from 'react'
import { cn } from './cn'

export default function Textarea({ className, ...props }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      className={cn(
        'block w-full rounded-md border border-chat-border bg-white text-chat-foreground placeholder:text-chat-muted-foreground',
        'px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-chat-primary focus:border-chat-primary',
        'disabled:opacity-60 disabled:cursor-not-allowed',
        'min-h-[60px] resize-y',
        className,
      )}
      {...props}
    />
  )
}

