# Meta-prompt — NegotiatorPro dialogue + right gutter

Hand this to a coding agent (or paste into a new Claude Code session) when ready to implement. The standalone mockup lives next to this file at `dialogue-gutter.html` — open it in a browser for the visual reference.

---

You are implementing a UI revision to NegotiatorPro, the active tool in the Amfonica platform. The codebase lives at `/Users/thomasoneill/Dev.local/works-in-progress/sales-partner/NegotiatorPro`. Frontend is React 18 + TypeScript on Vite, state in Zustand. Read `NegotiatorPro/CLAUDE.md` first, then `frontend/src/App.tsx` and `frontend/src/components/` to understand the current chat surface. Visual reference is `mockups-ui/dialogue-gutter.html` — open it before you write any code.

## What we're building

A three-pane layout that promotes the dialogue to the main view and adds a persistent, collapsible right-hand stats gutter.

1. **Left rail** — collapse the existing settings sidebar to a 44px icon strip. Add a toggle at the top of the strip (`ti-layout-sidebar-left-expand` / `ti-layout-sidebar-left-collapse`) that restores the full settings panel. Settings panel state persists across sessions.
2. **Chat** — the dialogue takes the bulk of the horizontal width and is the main view.
3. **Right gutter** — two states: collapsed (~240px, matches the left rail's collapsed width as a visual rhyme) and expanded (~420px). The chevron toggle sits on the gutter's leading edge. Expanded state can either push the chat narrower or overlay it — Matt to decide based on feel during implementation; build the push behaviour first and we'll evaluate.

## The gutter contents

**Collapsed gutter shows two blocks:**

- **PLEASE score** — six small radial gauges in a 3×2 grid. Total `N / 30` in the section header. See PLEASE spec below.
- **Sources in play** — list of recent citations. Active citation gets an amber left rail and shows a one-line quote from the relevant passage. Hover expands the passage.

**Expanded gutter adds:**

- A thin progress bar below the PLEASE header showing the `/30` total with a tick at 20 (the rewrite threshold). One-line read of the weakest letters.
- **Leverage** — horizontal split bar. Left = user, right = negotiation partner. Width shows where leverage sits; a centre tick marks parity. Below: two columns of inferred edges (user / partner). Inferred-with-low-confidence items render in muted grey rather than the strong colour.
- **Parties** — two compact cards, one per side. Avatar initial, name, one-line summary.
- **Vitals** — three small tiles: ZOPA, BATNA strength, time pressure.
- The Sources block stays at the bottom, same as the collapsed state.

## PLEASE framework (do not invent labels — these are the real ones)

A self-assessment scoring tool the backend applies to analysis-intent responses. Each letter is graded 1–5:

- **P** — Polite
- **L** — Logical
- **E** — Empathetic
- **A** — Assertive
- **S** — Strategic
- **E** — Engaging

Total out of 30. The system prompt instructs the model to revise any response scoring below 20. The UI must show the total `N / 30` and mark the 20 threshold visually.

Gauge colour by score (use one ramp consistently across the app — these are the values used in the mockup, swap for the project's tokens once palette work lands):

- 5 → `#0F6E56` deep teal
- 4 → `#1D9E75` teal
- 3 → `#BA7517` amber
- 2 → `#A32D2D` red
- 1 → deeper red

## Data contract

The backend needs to surface per-response scoring and per-negotiation context. Wire it up like this:

```ts
type PleaseScore = {
  polite: 1 | 2 | 3 | 4 | 5;
  logical: 1 | 2 | 3 | 4 | 5;
  empathetic: 1 | 2 | 3 | 4 | 5;
  assertive: 1 | 2 | 3 | 4 | 5;
  strategic: 1 | 2 | 3 | 4 | 5;
  engaging: 1 | 2 | 3 | 4 | 5;
  total: number;          // sum, 6–30
  weakest: string[];      // letter codes of the lowest scoring elements
};

type NegotiationContext = {
  please: PleaseScore;                // latest assistant response
  leverage: {
    mineWeight: number;               // 0–1, sums with theirsWeight to 1
    theirsWeight: number;
    mine: { label: string; confidence: 'high' | 'low' }[];
    theirs: { label: string; confidence: 'high' | 'low' }[];
    summary: string;                  // e.g. "slight tilt — you"
  };
  parties: {
    me: { name: string; initial: string; summary: string };
    them: { name: string; initial: string; summary: string };
  };
  vitals: {
    zopa: string | null;              // e.g. "£42–58k"
    batna: 'strong' | 'moderate' | 'weak' | null;
    time: 'urgent' | 'tight' | 'loose' | null;
  };
  sources: {
    title: string;                    // e.g. "Voss · Never Split the Difference"
    sub: string;                      // e.g. "Mirroring · ch. 3"
    quote?: string;                   // pulled passage for active source
    active: boolean;
  }[];
};
```

The PLEASE score already exists in the backend prompt machinery — extract it from the model's self-assessment block and expose it via the chat response payload. Leverage, parties, vitals and live sources will need new backend work — implement the frontend against a typed stub first, then wire the API.

## Design constraints

Follow `jsx-viewer/STYLEGUIDE.md` for tokens, type, and spacing. Reinforcing the parts most relevant here:

- Inter, system fallbacks.
- Accent only: `#A8C5FF`. No gradients, no purple, no neon.
- Borders `1px solid #E0E0E0`. Cards `12px` radius. Buttons / inputs `8px`.
- Sentence case throughout. No em dashes. No emoji. No business clichés.
- Two font weights: 400 and 500.
- Eyebrow labels (`PLEASE`, `LEVERAGE`, `SOURCES IN PLAY`) carry section identity. No heavy section headers.

## Interactions

- Right gutter chevron toggles collapsed ↔ expanded with a 200ms ease.
- Left rail toggle restores the full settings sidebar with the same timing.
- Citation chips in chat messages are hover-targeted. Hovering a chip highlights its corresponding entry in the Sources list (and vice versa).
- As the user scrolls past a chat message, its citations become the "active" Sources entry. This is the live highlight behaviour.
- All persistent UI state (sidebar collapsed, gutter collapsed, last-active negotiation) saves to the existing Zustand store and rehydrates on reload.

## What to build first

1. Pure presentational `RightGutter` component with `collapsed` prop, fed mocked data. Match the HTML mockup pixel-for-pixel using project tokens.
2. Left rail collapse / settings panel toggle. Wire to Zustand.
3. Wire PLEASE from the existing backend self-assessment. This is the highest-value live data and the easiest to surface.
4. Stub the rest of `NegotiationContext` in the store with sensible mock data. Verify the gutter renders properly.
5. Backend work for leverage, parties, vitals, live sources. One PR per block.

Don't bundle this into a single mega-PR. Land it in the order above.

## House style reminders

Plain Anglo-Saxon prose in any user-facing strings. No "leverage" the verb, no "unlock", no "deep dive". Code comments explain why, not what. No commented-out code. Type hints / TS types throughout.
