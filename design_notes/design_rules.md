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

## Implementation Order

1. **Theme switch** (Rule 1) — landing page goes light/white
2. **Typography** (Rule 2) — import serif font, size up the headline
3. **Navigation** (Rule 6) — simplify links, add pill container
4. **Hero content** (Rules 3, 4, 5) — simplify CTAs, max whitespace, single accent
5. **Product mockup** (Rule 8) — add screenshot/preview
6. **Feature cards** (Rule 9) — replace emojis, clean card design
7. **Shapes** (Rule 7) — pill buttons, rounded cards
8. **Trust bar** (Rule 10) — restyle tech section as horizontal strip
