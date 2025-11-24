import React from 'react';

type Variant = 'primary' | 'secondary' | 'danger' | 'outline' | 'ghost';
type Size = 'sm' | 'md' | 'lg';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

const variantClasses: Record<Variant, string> = {
  primary: 'bg-chat-primary text-white hover:bg-chat-primary-hover',
  secondary: 'bg-secondary text-white hover:bg-secondary/90',
  danger: 'bg-danger text-white hover:bg-red-700',
  outline: 'bg-transparent border border-border text-foreground hover:bg-muted',
  ghost: 'bg-transparent text-foreground hover:bg-muted',
};

const sizeClasses: Record<Size, string> = {
  sm: 'px-3 py-1.5 text-[13px] rounded',
  md: 'px-4 py-2 text-[14px] rounded-md',
  lg: 'px-5 py-3 text-[16px] rounded-lg',
};

export default function Button({
  variant = 'primary',
  size = 'md',
  className = '',
  disabled,
  ...props
}: ButtonProps) {
  const base = 'inline-flex items-center justify-center font-medium transition disabled:opacity-50 disabled:cursor-not-allowed';
  return (
    <button
      className={`${base} ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
      disabled={disabled}
      {...props}
    />
  );
}

