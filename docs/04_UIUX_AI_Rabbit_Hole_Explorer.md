# UI/UX Design Document
## AI Internet Rabbit-Hole Explorer

---

## 1. Design Principles
1. **The graph is the product.** Every other UI element (sidebars, panels, search) exists to support navigation of the graph — never compete with it visually.
2. **Progressive disclosure.** Don't show a 50-node graph at once. Start small (8–12 nodes), let curiosity drive expansion.
3. **Always show a "way out."** A learner should never feel lost in the rabbit hole — the learning path and breadcrumb are always visible.
4. **Low-friction reading.** Summaries default to the shortest useful form (2-min); deeper content is opt-in, not forced.
5. **Calm, focused visual language.** Muted background, high-contrast nodes, restrained color-coding (color communicates state, not decoration).

## 2. Information Architecture

```
Landing
 └── Graph Workspace (per topic)
      ├── Top Bar (topic title, semantic search, new topic)
      ├── Left Panel: Learning Path
      ├── Center: Interactive Graph Canvas
      ├── Right Drawer: Node Detail (Summary / Sources / Related)
      ├── Right Panel (toggle): Trends
      └── Right Panel (toggle): Progress
```

## 3. Key Screens — Layout Detail

### 3.1 Landing Page
- Centered layout, minimal chrome.
- Large search input with placeholder: "e.g., Artificial General Intelligence, Quantum Computing, Blockchain..."
- Below: 4–6 example topic chips.
- Subtle animated background graph (decorative, non-interactive) to preview the product visually.

### 3.2 Graph Workspace
- **Canvas:** occupies ~65–70% of viewport width.
  - Nodes: rounded pill/circle shapes, size scaled slightly by importance (degree centrality).
  - Edges: directional arrows for `prerequisite_of`/`enables`/`breaks`; undirected soft lines for `related_to`.
  - Edge labels appear on hover only (avoid clutter by default).
- **Left panel (Learning Path):** numbered vertical list, current node highlighted, completed nodes checked off, clicking an item pans/zooms the graph to that node.
- **Right drawer (Node Detail):** slides in from the right, does not cover the graph entirely (max 35% width on desktop), dismissible via close icon or clicking canvas background.

### 3.3 Node Detail Panel
- Header: node title + small tag row (e.g., "Core Concept," "Trending").
- Tab bar: Summary | Sources | Related
- Summary tab: segmented control (2-min / 10-min / Deep-dive), text rendered with inline citation markers `[1] [2]` linking to the Sources tab.
- Sources tab: card list, each card shows source-type icon (Wikipedia / YouTube / Reddit / Paper / Web), title, and a one-line snippet.
- Sticky footer: "Mark as Learned" (primary), "Expand Further" (secondary).

### 3.4 YouTube Extraction Modal
- Split view: left = embedded video player, right = tabs (Notes | Key Concepts | Quiz).
- Quiz tab: simple radio-button questions, immediate inline feedback on submit, no page reload.

### 3.5 Trends Panel
- List of trend cards: topic name, sparkline of paper-count growth, % growth badge, "Add to graph" button.

### 3.6 Progress Panel
- Circular progress indicator (% of graph explored).
- List of learned nodes (chips).
- "Suggested next" card with a single clear CTA.

## 4. Visual Design System

### 4.1 Color Palette (semantic use)
| Token | Use |
|---|---|
| `--bg-canvas` | Deep neutral (near-black or off-white depending on theme) — graph background |
| `--node-default` | Neutral slate — unexplored node |
| `--node-inprogress` | Amber accent border |
| `--node-learned` | Muted green fill, checkmark icon |
| `--node-selected` | Bright accent glow (brand primary color) |
| `--edge-prereq` | Solid, medium contrast, arrowed |
| `--edge-related` | Dashed, low contrast |
| `--trend-badge` | Warm accent (orange/red) to signal "hot" topics |

Support both light and dark themes — graph exploration sessions are often long, dark mode reduces eye strain.

### 4.2 Typography
- Sans-serif throughout for UI chrome (e.g., Inter or system font stack).
- Node labels: medium weight, truncated with ellipsis + full label on hover tooltip.
- Summary text: comfortable reading line-height (1.6), max-width constrained (~65ch) even inside the drawer for readability.

### 4.3 Motion
- Node expansion: new nodes fade/scale in and animate from their parent node's position (spatial continuity so the user doesn't lose orientation).
- Panel transitions: slide, 200–250ms ease-out, no bounce (keeps the tool feeling precise/technical rather than playful).
- Avoid animating the whole graph layout on every interaction — only affected subgraph should reflow.

## 5. Interaction Patterns
- **Hover:** node hover shows a lightweight tooltip (title + one-line description) before committing to a click.
- **Click:** opens detail drawer without altering graph layout.
- **Double-click / Expand button:** triggers rabbit-hole expansion.
- **Drag:** manual repositioning, persisted per user.
- **Keyboard:** `/` focuses search-within-graph; `Esc` closes any open panel; arrow keys move focus between learning-path items.

## 6. Accessibility
- All node states communicated with both color AND icon/shape (not color alone) for color-blind users.
- Full keyboard navigation for the learning path list and node detail panel.
- ARIA live region announces when new nodes are added to the graph (important since expansion happens asynchronously).
- Minimum contrast ratio 4.5:1 for all text.

## 7. Responsive Behavior
- **Desktop (primary target):** full 3-pane layout as described above.
- **Tablet:** right drawer becomes a full-height overlay instead of a fixed-width panel; left learning-path panel collapses into a toggleable drawer.
- **Mobile:** graph canvas becomes primary full-screen view; learning path and node detail become bottom-sheet modals; expansion/summary interactions unchanged, just re-flowed vertically.

## 8. Empty / Loading / Error States
- **Empty (first load):** decorative animated graph + prompt to enter a topic.
- **Loading (per stage):** skeleton nodes pulse in placeholder positions while real data streams in.
- **Error (source failure):** small non-blocking toast, graph continues rendering with available data.
- **Graph too large:** contextual banner: "This topic has grown large — try Focus Mode" with one-click toggle.
