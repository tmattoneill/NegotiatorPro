import React from 'react';

type SelectProps = React.SelectHTMLAttributes<HTMLSelectElement>;

export default function Select({ className = '', children, ...props }: SelectProps) {
  const base = 'w-full px-3 py-2 border border-border rounded text-[14px] bg-white text-foreground outline-none focus:border-primary focus:ring-2 focus:ring-primary/10';
  return (
    <select className={`${base} ${className}`} {...props}>
      {children}
    </select>
  );
}

