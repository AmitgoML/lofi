# Creative Director Agent — Flow Documentation

## What It Does

The Creative Director Agent sits between the **Performance Analyst** and the final **QA / Proposal Assembly** steps. It takes performance insights + brand data and produces three creative outputs:

| Slot | What it is | How it's produced |
|------|-----------|-------------------|
| **Slot 1 — Recommended** | Best-performing historical asset | Surfaced from PerformanceAnalyst — no generation, just the `asset_id` reference |
| **Slot 2 — Variant A** | Brand-hero image | Generated fresh by Titan Image Generator v2 |
| **Slot 3 — Variant B** | Lifestyle/context image | Generated fresh by Titan Image Generator v2 |

The agent also generates ad copy (headlines, descriptions, CTA, hooks) and determines the overall creative strategy.

---

## Flow Step-by-Step

### 1. Receive State from Performance Analyst

The workflow state carries a `PerformanceAnalystOutput` that includes:
- `brand_id` — UUID of the brand looked up from the `brands` table
- `creative_recommendations` — ranked list of creative formats with the best-performing `asset_id` per format
- `platform_recommendations`, `audience_recommendations`, `location_recommendations`

```
WorkflowState
  └── performance_insights: PerformanceAnalystOutput
        ├── brand_id: "8cfb9676-..."
        └── creative_recommendations:
              └── [{ format: IMAGE, asset_id: "527c0bdc-...", engagement_rate: 0.04 }]
```

---

### 2. Fetch Full Brand Data from Supabase

Using the `brand_id`, the agent calls `SupabaseClient.get_brand()` to pull the full brand row from the `brands` table.

Fields fetched for **visual identity** (used by image generator strategy):
- `brand_primary_color`, `brand_secondary_color`
- `brand_imagery_style`, `brand_design_elements`
- `brand_logos`, `brand_reference_images`
- `brand_heading_font`, `brand_body_font`
- `brand_logo_usage_rules`, `brand_dos_and_donts`

Fields fetched for **brand voice** (used by copywriter):
- `brand_tone_of_voice`, `brand_copywriting_tone`
- `brand_core_values`, `brand_messaging_pillars`
- `brand_tagline`, `brand_positioning`
- `brand_keyword_blacklist`, `brand_competitors`
- `brand_audiences`, `product_descriptions`, `brand_goal_config`

All fields are mapped into a `BrandRow` Pydantic model (`persistence/models.py`).

---

### 3. Determine Creative Strategy (Claude Sonnet via Bedrock)

The agent sends a detailed prompt to **Claude Sonnet 4.5** (via `BedrockClient.extract_structured`) asking it to output a structured `CreativeStrategy`.

The prompt (`prompts/creative_director.py :: creative_strategy_v1`) includes:
- Full brand profile (colors, imagery style, logo rules, dos/don'ts)
- Campaign brief (goal, platforms, audience, offer)
- Historical performance insights (top platforms, formats, audience segments)
- Existing assets available for reuse

Claude is forced via **tool-use** to return a structured `CreativeStrategy` object. No free-text parsing needed.

#### CreativeStrategy output:
```
CreativeStrategy
  ├── best_creative_format: IMAGE
  ├── best_messaging_angle: "..."
  ├── asset_decisions: [{ format: IMAGE, action: "generate" }]
  ├── variant_a: ABVariant
  │     ├── variant_label: "A"
  │     ├── image_prompt: "Bold product shot, #0057FF background, logo top-right..."
  │     ├── negative_prompt: "blurry, text overlay, watermark..."
  │     └── rationale: "Brand-hero: product as focal point, drives conversion"
  ├── variant_b: ABVariant
  │     ├── variant_label: "B"
  │     ├── image_prompt: "Team collaborating in modern office, natural light..."
  │     ├── negative_prompt: "blurry, staged-looking..."
  │     └── rationale: "Lifestyle: emotional resonance, upper-funnel awareness"
  └── rationale: "Test brand-direct vs aspiration to learn audience response"
```

#### Why two variants?

Real A/B testing in paid media tests one variable at a time. The two variants are deliberately architected to test different hypotheses:

- **Variant A — "Brand Hero"**: Product or service is the unmistakable visual focus. Brand colors dominate. Logo placed prominently (top-right per brand guidelines). Clean, structured layout — designed to convert users who already know the category.

- **Variant B — "Lifestyle / Context"**: Product shown in a real-world or aspirational setting. Scene-first composition. Brand colors appear as accents. Logo subtle (bottom-left, watermark-style). More immersive, less ad-like — builds brand affinity and upper-funnel recall.

---

### 4. Generate Images — Variant A and B

The `ImageGeneratorAgent` is called **twice** with the prompts from Claude's strategy.

Model: `amazon.titan-image-generator-v2:0` via **AWS Bedrock**.

For each variant, one image is generated **per platform** in the campaign brief:

| Platform | Dimensions |
|----------|-----------|
| META | 1024 × 1024 (1:1 square) |
| TIKTOK | 576 × 1024 (~9:16 vertical) |
| GOOGLE | 1024 × 576 (~16:9 horizontal) |
| SPOTIFY | 640 × 640 (1:1 square) |

All dimensions are multiples of 64 (Titan requirement, range 320–4096px).

Each generated image is uploaded to **S3** immediately:
```
s3://<S3_BUCKET>/creatives/<brand_name>/<platform>/uuid.png
```
Example:
```
s3://lofi-creatives/creatives/Acme Corp/meta/3f4a8b2c-1234-....png
s3://lofi-creatives/creatives/Acme Corp/google/7d9e1f3a-5678-....png
```

---

### 5. Generate Copy (Claude Sonnet via Bedrock)

The `CopywriterAgent` calls Claude Sonnet using `BedrockClient.extract_structured` with a brand-voice-aware prompt (`prompts/copywriter.py :: copy_generation_v1`).

The prompt uses all the brand voice fields to ensure copy sounds like the brand, not generic ad copy. It explicitly references:
- Tone of voice + copywriting tone
- Messaging pillars, tagline, positioning
- Prohibited keywords (blacklist)
- Competitor brands to differentiate from

Output is a structured `TextAsset`:
```
TextAsset
  ├── headlines: ["Build faster", "Less busywork", ...]   (5, max 30 chars each)
  ├── descriptions: ["Acme automates the work...", ...]   (3, max 90 chars each)
  ├── cta: "Start Pilot"                                  (max 15 chars)
  ├── hooks: ["Tired of manual handoffs?", ...]           (3, max 50 chars each)
  ├── keywords: ["workflow automation", ...]              (5 targeting keywords)
  └── long_headlines: ["Build better products faster...", ...] (2, max 90 chars)
```

---

### 6. Output

The agent assembles a `CreativeDirectorOutput` written to `state["creative_director_output"]`:

```
CreativeDirectorOutput
  ├── recommended_assets: [RecommendedAsset]   ← Slot 1: historical best (no S3, just asset_id)
  ├── variant_a: [AssetRef]                     ← Slot 2: brand-hero S3 URLs per platform
  ├── variant_b: [AssetRef]                     ← Slot 3: lifestyle S3 URLs per platform
  ├── best_creative_format: IMAGE
  ├── best_messaging_angle: "..."
  ├── variant_a_rationale: "Brand-hero: ..."
  ├── variant_b_rationale: "Lifestyle: ..."
  ├── asset_decisions: [AssetDecision]
  └── texts: TextAsset
```

---

## Key Files

| File | Purpose |
|------|---------|
| `agents/creative_director.py` | Main agent — orchestrates the full flow |
| `agents/sub_agents/image_generator.py` | Calls Titan v2, uploads to S3 |
| `agents/sub_agents/copywriter.py` | Calls Claude Sonnet for ad copy |
| `schemas/creative_director.py` | All input/output Pydantic models |
| `persistence/models.py` | `BrandRow` — mirrors the `brands` DB table |
| `persistence/supabase_client.py` | `get_brand()` fetches brand data |
| `persistence/s3_storage.py` | `upload_asset()` pushes images to S3 |
| `prompts/creative_director.py` | `creative_strategy_v1()` — Claude strategy prompt |
| `prompts/copywriter.py` | `copy_generation_v1()` — Claude copy prompt |
| `prompts/image_generator.py` | `DEFAULT_NEGATIVE_PROMPT_V1` constant |

---

## Environment Variables Required

```env
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=arn:aws:bedrock:us-east-1:034362062829:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0
IMAGE_MODEL_ID=amazon.nova-canvas-v1:0
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_KEY=<supabase-service-role-key>
S3_BUCKET=<your-s3-bucket-name>
```
