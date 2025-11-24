import React from 'react';

type InputProps = React.InputHTMLAttributes<HTMLInputElement>;

export default function Input({ className = '', ...props }: InputProps) {
  const base = 'w-full px-3 py-2 border border-border rounded text-[14px] outline-none focus:border-primary focus:ring-2 focus:ring-primary/10';
  return <input className={`${base} ${className}`} {...props} />;
}

