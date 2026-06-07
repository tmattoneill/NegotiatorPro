# Palette starters

A library of full palette exports. None of these is imported by the app. The
live theme is **`../palette.css`**, which the app pulls in via `index.css`:

```css
@import './styles/palette.css';
```

To re-theme the whole app, copy a starter over the live file:

```bash
cp src/styles/palette-starters/matrix.css src/styles/palette.css
```

That's the only step. Every colour in the UI routes through the semantic
aliases these files define, so nothing else changes.

## What a starter contains

Two layers, in this order:

1. **Raw ramps** — `--palette-dominant-1..6` (lightest to darkest neutral) and
   `--palette-secondary-1..3` (lightest to darkest accent), plus
   `--palette-highlight`. Components must never reference these directly.
2. **Semantic aliases** — `--color-background`, `--color-surface`,
   `--color-body`, `--color-heading`, `--color-accent`, `--color-accent-strong`,
   `--color-pop`, etc. Components reference *only* these. The `[data-theme="dark"]`
   block re-points the dominant aliases down the ramp (background → darkest,
   heading → lightest); the accent aliases stay put across themes.

This is exactly what [site-palette](https://site-palette) emits.

## The `on-*` (foreground) tokens

Newer exports emit an "on" role for each backgroundable colour — a foreground
guaranteed to contrast with that fill:

```css
--color-on-accent:        var(--palette-dominant-6);  /* pairs with --color-accent (mid) */
--color-on-accent-soft:   var(--palette-dominant-6);
--color-on-accent-strong: var(--palette-dominant-1);  /* pairs with --color-accent-strong (dark) */
--color-on-pop:           var(--palette-dominant-1);
```

The contract is **suffix pairing**: a fill of `--color-X` takes foreground
`--color-on-X`. Mind the binding — `on-accent` pairs with the *mid* accent and
can be dark (dominant-6 above), so it is the wrong token for a dark
`accent-strong` fill. The user bubble fills with `accent-strong`, so it uses
`on-accent-strong`.

Older exports (the four starters here) predate these tokens. The app handles
both with a fallback, in `index.css`, **not** in the palette files:

```css
--user-bubble-fg: var(--color-on-accent-strong, var(--palette-dominant-1));
```

Prefer the palette's paired token when present; fall back to the lightest
neutral otherwise. That fallback lives outside the palette files on purpose, so
a raw export swaps in wholesale and the bubble stays readable either way. It
does not flip with dark mode, because the accent fill it sits on doesn't flip.

## Contrast contract

The chat bubbles depend on one structural invariant that every starter must hold:

- `--color-accent-strong` is the **dark** end of the accent ramp (low L).
- the bubble foreground resolves to the **light** end (`on-accent-strong` =
  `--palette-dominant-1`, or the dominant-1 fallback for older exports).

The user bubble fills with `accent-strong` and writes in `--user-bubble-fg`; the
assistant fills with `surface` and writes in `body`/`heading`. As long as the
ramps run light→dark in the documented order, every pairing clears WCAG AA
(≥ 4.5:1) in both themes. A mid-tone accent fill (`--color-accent`, ~L 62%)
can't be rescued by any text colour — that was the original bug.

## Swap-regression results

Measured with true sRGB sampling (headless Chrome canvas) + WCAG maths. Worst
case across all four palettes × both themes is **4.51** — all pass AA.

| Palette | Theme | user | asst body | asst head |
|---|---|---|---|---|
| Sand + Blue | light | 8.85 | 7.00 | 12.64 |
| Sand + Blue | dark | 8.85 | 7.00 | 8.98 |
| Gold + Teal | light | 8.71 | 6.97 | 12.63 |
| Gold + Teal | dark | 8.71 | 6.97 | 8.94 |
| Mad Max | light | 9.34 | 7.11 | 12.71 |
| Mad Max | dark | 9.34 | 7.11 | 9.08 |
| Matrix | light | 8.98 | 6.83 | 13.09 |
| Matrix | dark | 8.98 | 6.83 | 8.74 |
| Pastel STRESS¹ | light/dark | 4.51 | 7.00 | 12.64 |

¹ A deliberately hostile case: accent-strong forced to L 55% (much lighter than
any real export). It still clears AA, which marks the boundary — if a swapped
palette ever pushes accent-strong above ~L 60%, re-check the user-bubble ratio.

## Notes on the alternates

`mad-max.css` and `matrix.css` were authored in site-palette's format for this
regression test, not exported from the tool (site-palette isn't wired into this
repo). They follow the export contract exactly, so a real export can replace
them at any time. `sand-blue.css` and `gold-blue.css` are genuine exports.
