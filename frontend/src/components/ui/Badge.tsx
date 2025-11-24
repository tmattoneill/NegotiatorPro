import React from 'react';

type Variant = 'default' | 'success' | 'warning' | 'danger' | 'neutral' | 'primary';

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: Variant;
}

const variantClasses: Record<Variant, string> = {
  default: 'bg-muted text-foreground',
  primary: 'bg-chat-primary text-white',
  success: 'bg-success text-white',
  warning: 'bg-warning text-white',
  danger: 'bg-danger text-white',
  neutral: 'bg-gray-400 text-white',
};

export default function Badge({ variant = 'default', className = '', ...props }: BadgeProps) {
  const base = 'inline-flex items-center px-2 py-0.5 rounded-full text-[12px] font-medium';
  return <span className={`${base} ${variantClasses[variant]} ${className}`} {...props} />;
}

