# SignMate Frontend Redesign Rules

Transform the current frontend to match the design language seen across all competitor landing pages.

---

## Rule 1 — Switch to a Light/White Theme for the Landing Page

**What competitors do**: 4 out of 5 use a white or off-white background. Clean, open, maximizes readability.

**What to do**:

- Set the homepage background to **white or off-white** (`#fafbfc` / `#ffffff`)
- All text becomes **dark charcoal/black** for headlines, **medium grey** for body
- Remove glassmorphism, dark cards, and translucent overlays from the landing page
- The dark theme can remain for the in-app pages (tracker, transcriber, etc.) — landing page should feel distinct

---

## Rule 2 — Large Serif Headline as the Hero

**What competitors do**: 4 of 5 use bold serif fonts at 48–150px for the hero headline. It's the dominant visual element.

**What to do**:

- Import a **serif display font** (e.g., `'Playfair Display'`, `'Source Serif 4'`, or `'Instrument Serif'` from Google Fonts)
- Set the hero headline to **56–72px, bold, serif**, dark/black color
- Keep `DM Sans` for all body text and navigation
- The headline should be **the largest visual element** on the entire page — nothing else comes close

---

## Rule 3 — Minimal Hero Content with Clear Vertical Hierarchy

**What competitors do**: Badge → Headline → 1-2 line subtext → 1-2 CTA buttons. Center-aligned. Nothing else.

**What to do**:

- Remove the 3-button CTA grid. Replace with **1 primary filled button** + **1 secondary outlined/text button**
- Primary: "Start Translating →" (filled, accent green, pill-shaped)
- Secondary: "Learn More" or "See How It Works" (outlined or text link)
- Shorten the hero description to **1–2 sentences max**
- Keep the typing animation for the title — it's a unique differentiator
- Center-align everything vertically: title → description → CTAs

---

## Rule 4 — Extreme Whitespace

**What competitors do**: Hero sections have 120–250px of vertical padding. Content floats in open space.

**What to do**:

- Hero top padding: **140–180px** (from nav), bottom padding: **100–120px**
- Gap between hero elements: **24–32px**
- Gap between hero and next section: **100–120px**
- Max-width for hero content: **720–800px**, centered
- Let the page **breathe** — remove anything that makes sections feel cramped

---

## Rule 5 — Single Accent Color

**What competitors do**: Monochrome base + exactly 1 accent color, used almost exclusively on CTAs.

**What to do**:

- **Emerald green** remains the sole accent — only appears on:
  - Primary CTA button fill
  - Hover states
  - Active/selected indicators
- Hero headline: **black/dark**, not green
- Feature icons: **dark or grey**, not colored emoji
- Everything else: **black, white, and greys only** on the landing page

---

## Rule 6 — Streamlined Navigation

**What competitors do**: Logo left, 3–4 clean text links, 1 CTA button on the right. Minimal and unobtrusive.

**What to do**:

- Reduce nav to **3–4 links max**: Translator | Guide | Practice (or similar core pages)
- Move secondary pages (History, Quiz, Transcriber) to the footer only
- Right side: a **single pill-shaped CTA button** ("Get Started" or "Try Free")
- Settings → **icon-only**, no label
- Consider wrapping nav links in a **pill container with subtle border** (like Dovetail, Ctrl)
- Nav background: **white/transparent**, not dark — text in dark grey/black

---

## Rule 7 — Pill-Shaped / Rounded Everything

**What competitors do**: All interactive elements use soft rounded shapes. No sharp corners.

**What to do**:

- CTA buttons: **full pill radius** (`border-radius: 9999px`)
- Nav container: **pill-shaped** with subtle 1px border
- Feature cards: **20–24px border-radius**
- All badges, tags, status indicators: **pill-shaped**
- Remove any square-cornered elements from the landing page

---

## Rule 8 — Add a Product Visual / Mockup

**What competitors do**: Every competitor shows their product — device mockups, UI screenshots, or app previews.

**What to do**:

- Add a **product screenshot** or **demo preview** between the hero and the features section
- Show the tracker/translator in action — a screenshot wrapped in a rounded card with shadow
- This is the **single most impactful addition** — competitors all "show, don't tell"
- Optional: add a subtle **colored glow/shadow** behind the preview (like Popcorn's warm glow)

---

## Rule 9 — Clean Feature Cards with Proper Icons

**What competitors do**: Icon + title + short description in a 3-column grid. Icons are custom/SVG, not emoji.

**What to do**:

- Replace emoji icons (📹, ✋, 💾, 📊) with a **consistent SVG icon set** (Lucide, Phosphor, or Heroicons)
- Keep the grid layout but ensure cards have:
  - **24–32px internal padding**
  - **Subtle border** or light background (no heavy glassmorphism)
  - **2 lines max** for descriptions
- Optional: add a **subtle colored top-edge accent** on each card (like Lens Protocol)
- Cards on white background with light shadows or light grey background fill

---

## Rule 10 — Restyle the Tech/Trust Section

**What competitors do**: Horizontal logo bar or trust strip — muted, subtle, not a big section.

**What to do**:

- Convert the current vertical "Built With" grid into a **horizontal inline strip**
- Use small, monochrome/grey tech logos or icons in a single row
- Small label above: "Powered By" or "Built With"
- Visually subtle — this should feel like a footnote, not a section

---

## Rule 11 — Add an FAQ Section

**What competitors do**: Both Shuttle and the dark SaaS use centered accordion-style FAQ sections with expandable questions. Common patterns:

- Large bold title ("FAQ") centered above the accordion
- Optional subtitle or badge above the title
- Accordion items: question text + expand/collapse icon (`v` chevron or `+`/`−`)
- Only one item expanded at a time
- Narrow, centered container (~650px max-width)
- Clean separation between items (subtle borders or gaps)

**What to do**:

- Add an **FAQ section** between the trust bar and the footer
- Center a **large bold title** ("Frequently Asked Questions") matching our sans-serif display font
- Use **white accordion cards** on the off-white homepage background (matching Shuttle's light pattern)
- Each item: question on the left, **chevron or plus/minus icon** on the right
- **Click to expand/collapse** — only one open at a time
- Content: 5–6 questions relevant to SignMate (what it does, privacy, accuracy, requirements, etc.)
- Generous internal card padding (20–24px), rounded corners (12–16px)
- Keep the narrow centered layout (~700px max-width)

---

## Rule 12 — Redesign the "How It Works" Section

**What competitors do**: Across 4 competitor "How it works" sections (fintech receipt tool, Framer Teams, GitBook ×2), several strong patterns repeat:

- **Two-column split layout** — 3 of 4 competitors use left context + right content (or left steps + right product visual)
- **Explicit step numbering** — bold numerals (01, 02, 03…) or colored icons to anchor each step
- **Product visual as proof** — a real UI mockup alongside the steps, synced to the active step or used as a centerpiece
- **Extreme whitespace** — 80–120px section padding, 40–60px between steps
- **Single accent color on step markers only** — never on text or backgrounds
- **Minimal containers** — spacing does the structural work; if a container exists, it's 1px border with near-transparent fill
- **3-tier type hierarchy** — massive section headline → medium step titles → small descriptions

**What to do**:

### Layout

- Replace the current **3-card grid** with a **two-column split layout**:
  - **Left column (~35–40%)**: section title ("How It Works"), a 1–2 line subtitle describing the pipeline, and optionally a CTA button
  - **Right column (~60–65%)**: the step list, stacked vertically
- Alternative layout option: **numbered horizontal card row** (like GitBook's 01–05 pattern) — works if limiting to 4–5 steps

### Step Content

- Expand from **3 steps to 6 steps** covering the full pipeline (see `how_it_works_expanded_draft.md`):
  1. Camera Capture
  2. Hand Detection & Landmark Tracking
  3. Feature Extraction & Normalization
  4. AI Classification
  5. Word & Sentence Building
  6. Full-Word Recognition (Enhanced Mode)

### Step Numbering & Icons

- Add **explicit step numbers** (01, 02, 03…) in a bold, larger font — use the **emerald accent color** for numbers only
- Pair each number with a **small monochrome SVG icon** (from Lucide or similar) inside a subtle rounded-square container with light fill
- Icons should be **20–24px**, line-style, matching existing icon set

### Typography within the Section

- **Section headline**: 32–40px, bold, dark — same style as other section titles
- **Optional subtitle**: 15–16px, regular, medium grey, 1–2 lines
- **Step titles**: 18–22px, bold/semibold, dark
- **Step descriptions**: 14–15px, regular, medium-dark grey, **2–3 lines max** — user-facing, non-technical language

### Spacing & Containers

- Section top/bottom padding: **100–120px**
- Between steps: **40–60px** vertical gap (if vertical list)
- If using cards: internal padding **24–32px**, **very subtle 1px border**, light off-white fill, **16–20px border-radius**
- No heavy shadows or dark backgrounds — keep it consistent with the light homepage theme

### Optional Enhancements

- **Product visual strip**: add a narrow product screenshot or animated SVG diagram showing the pipeline (camera → landmarks → AI → text) between the steps and the trust bar
- **Stats strip** below the steps: "21 landmarks tracked · <30ms latency · 36 signs recognized · 370k word dictionary" — small, muted, inline

---

## Implementation Order

1. **Theme switch** (Rule 1) — landing page goes light/white
2. **Typography** (Rule 2) — import serif font, size up the headline
3. **Navigation** (Rule 6) — simplify links, add pill container
4. **Hero content** (Rules 3, 4, 5) — simplify CTAs, max whitespace, single accent
5. **Product mockup** (Rule 8) — add screenshot/preview
6. **Feature cards** (Rule 9) — replace emojis, clean card design
7. **Shapes** (Rule 7) — pill buttons, rounded cards
8. **Trust bar** (Rule 10) — restyle tech section as horizontal strip
9. **How It Works redesign** (Rule 12) — expand to 6 steps with split layout and numbering
10. **FAQ section** (Rule 11) — accordion FAQ between trust bar and footer
