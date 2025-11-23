declare module 'remarkable' {
  export interface RemarkableOptions {
    html?: boolean;
    breaks?: boolean;
    [key: string]: unknown;
  }

  export class Remarkable {
    constructor(options?: RemarkableOptions);
    render(markdown: string): string;
  }
}
