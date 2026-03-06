# How It Works — Common Patterns Across Competitors

Analysis of 4 competitor "How it works" sections, distilling the shared design principles.

---

## 1. Extreme Typographic Hierarchy

Every competitor uses at least 3 clearly distinct type tiers:

| Tier | Role | Size Range | Weight | Color |
|---|---|---|---|---|
| **Section label / tag** | Category identifier above the headline | 11–13px, caps, letter-spaced | Medium / Regular | Accent or muted grey |
| **Section headline** | The section's main statement | 36–48px | Bold | Highest contrast (white on dark, black on light) |
| **Step/card title** | Individual step heading | 16–24px | Bold / Semibold | High contrast |
| **Body / description** | Supporting paragraph per step | 13–15px | Regular | Muted / lower contrast grey |

**Key observation**: The jump between headline and body text is dramatic — at least 2.5× size difference. Step titles sit in between.

---

## 2. Single Accent Color Used with Extreme Restraint

All 4 competitors limit accent color to **one hue**, applied to only one or two element types:

- **Comp 1 (Fintech dark)**: no accent at all — purely monochrome
- **Comp 2 (Framer)**: bright blue on step icons only
- **Comp 3 (GitBook steps)**: orange/coral on step numbers and section label only
- **Comp 4 (GitBook features)**: orange confined entirely to the central illustration

**Pattern**: Accent never touches body text, headings, or backgrounds. It marks **navigation anchors** (icons, numbers) or **illustrative elements** (visuals).

---

## 3. Split / Two-Column Layouts Dominate

3 of 4 competitors use a **two-column layout** for this section:

| Competitor | Left Column | Right Column |
|---|---|---|
| Comp 1 (Fintech) | Step list with progressive disclosure | Product UI mockup |
| Comp 2 (Framer) | Section title + subtitle + CTA | Step list with icons |
| Comp 4 (GitBook features) | 2 feature cards | Central visual + 2 feature cards |

Only Comp 3 (GitBook steps) uses a **centered, single-column layout** with a horizontal card row.

**Pattern**: The left column provides context (heading, subtitle) or steps; the right column provides proof (mockup, visual) or the step content.

---

## 4. Product Visuals / Mockups as Proof

3 of 4 competitors include a **product visual** alongside the steps:

- **Comp 1**: full product UI mockup (receipt inbox) synced to the active step
- **Comp 4**: rich 3D-rendered product illustration as the centerpiece
- **Comp 2**: no visual, but the steps describe a clear process that implies the product

**Pattern**: "Show, don't tell." The product mockup validates the steps, turning abstract descriptions into concrete evidence.

---

## 5. Generous (Almost Excessive) Whitespace

Every competitor uses far more spacing than feels "necessary":

- Section padding: **80–120px** top and bottom
- Between steps: **40–100px** vertical gap
- Internal card padding: **24–32px**
- Max content width: centered and constrained (600–900px)

**Pattern**: Whitespace is the primary layout tool. No competitor relies on lines, dividers, or heavy borders to separate content. Spacing alone creates structure.

---

## 6. Progressive Disclosure or Clear Sequencing

Steps are presented with one of two sequencing strategies:

| Strategy | Used By | How |
|---|---|---|
| **Progressive disclosure** | Comp 1 (Fintech) | Only one step expanded; others are collapsed/greyed |
| **Explicit numbering** | Comp 3 (GitBook steps) | Large numbered labels (01–05) before each title |
| **Vertical ordering** | Comp 2 (Framer) | No numbers — top-to-bottom order + icons imply sequence |
| **No sequence (flat)** | Comp 4 (GitBook features) | All items equal weight, no order implied |

**Pattern**: When the section is a true "How it works" flow, either numbers or progressive disclosure is used. When it's a feature showcase, items are equal-weight.

---

## 7. Minimalist Containers

Cards and containers are either **invisible** (pure spacing) or extremely subtle:

- **Comp 1**: no containers at all — just text with spacing
- **Comp 2**: no containers — icons, headings, and descriptions float in space
- **Comp 3**: single unified container with subtle 1px border and light fill
- **Comp 4**: no containers around features — only the central visual has visual weight

**Pattern**: Avoid heavy card borders, shadows, or background fills. If a container exists, it's **one thin border** and **nearly transparent fill**.

---

## 8. Icons Are Small, Monochrome, and Contained

When icons are present, they follow strict rules:

- **Size**: 20–28px, never large or illustrative
- **Style**: monochrome line icons (single color, thin stroke) OR solid-filled circles with white symbol
- **Container**: either bare or inside a small rounded square with light background
- **Placement**: always flush-left or above the step title — never centered or decorative

---

## 9. CTA Placement

- **Comp 2 (Framer)**: CTA button inside the left context column (alongside heading)
- **Comp 4 (GitBook)**: CTA button under the headline, above the features
- **Comp 1 & 3**: no CTA within the section

**Pattern**: When a CTA exists, it's attached to the **context/heading area**, not appended after all the steps.

---

## Summary of Strongest Patterns to Adopt

1. **Two-column layout**: title/context on the left, step content on the right (or vice versa)
2. **Product visual as proof**: show the actual app UI alongside the explanation
3. **Explicit step numbering**: makes the pipeline feel concrete and sequential
4. **Single accent color**: used only on step numbers/icons, nothing else
5. **Extreme whitespace**: 80–120px between sections, 40–60px between steps
6. **Minimal containers**: spacing > borders > shadows
7. **3-tier type hierarchy**: massive headline → medium step titles → small descriptions
