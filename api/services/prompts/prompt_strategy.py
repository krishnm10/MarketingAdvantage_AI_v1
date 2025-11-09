from api.services.llm_connector import dual_chain_generate
from api.services.web_search import fetch_relevant_info
import random


def generate_strategy_prompt(
    business_name: str,
    industry: str,
    goal: str,
    specialization: str = "data_driven",
    temperature: float = 0.75,
    use_cache: bool = True,
    enable_web_search: bool = True
):
    """
    Generate a next-generation marketing strategy blueprint.
    Combines:
    - Dual LLM Chain (LLaMA → DeepSeek)
    - Human + AI intelligence
    - Psychological triggers
    - Strategic frameworks (RACE, Ansoff Matrix, STP, Data-driven Planning)
    - Structured JSON output
    """

    variation_seed = random.randint(1000, 9999)
    specialization = specialization.replace("_", " ").lower()

    tones = [
        "data-driven and ROI-focused",
        "psychologically intelligent and human-centered",
        "adaptive and growth-oriented",
        "creative yet analytical",
        "AI-augmented with strategic precision"
    ]
    tone = random.choice(tones)

    # Optional live market enrichment
    web_context = ""
    if enable_web_search:
        try:
            query = f"{industry} {specialization} marketing strategy trends 2025"
            web_context = fetch_relevant_info(query)
        except Exception:
            web_context = ""

    # Advanced Strategy Prompt
    prompt = f"""
(Strategy Variation #{variation_seed})

Design a comprehensive **marketing strategy blueprint** for '{business_name}' in the **{industry}** industry.
Primary Goal: {goal}
Focus: {specialization}
Tone: {tone}

This strategy must combine **human psychology**, **marketing science**, and **AI intelligence**.
It should include tactical insights guided by **four high-level strategic frameworks**:

---

🧩 1️⃣ Market Positioning & Differentiation (STP & UVP Framework)
- Define the unique value proposition (UVP): What makes this brand truly different?
- Identify key market segments and target audiences (Segmentation & Targeting).
- Articulate how to position the brand in customers’ minds — emotionally and rationally.
- Use Porter's Generic Strategies (Cost Leadership, Differentiation, Focus) to justify positioning.

---

📈 2️⃣ The Growth Vector (Ansoff Matrix)
- Apply the four growth paths:
  • Market Penetration (same product, same market)
  • Product Development (new product, same market)
  • Market Development (same product, new market)
  • Diversification (new product, new market)
- Specify which growth paths are most viable for this business and why.

---

🗺️ 3️⃣ The Customer Journey Framework (RACE: Reach, Act, Convert, Engage)
- Map every stage of the customer lifecycle:
  • Reach — building awareness through media and SEO.
  • Act — generating interactions and lead magnet engagement.
  • Convert — optimizing sales funnels, offers, and conversion events.
  • Engage — nurturing loyalty and advocacy post-purchase.
- Identify bottlenecks and how to optimize each phase.

---

📊 4️⃣ Data-Driven and Adaptive Strategy (The Planning)
- Integrate data-backed insights using SWOT and PESTLE analysis.
- Define SMART goals and performance metrics (KPIs).
- Build adaptive feedback loops for campaign iteration and resource reallocation.

---

🧠 Psychological Layer
- Embed human behavioral triggers (scarcity, reciprocity, social proof, authority, emotional resonance).
- Explain how storytelling, empathy, and visual cues influence decision-making.
- Include subconscious influence methods (color theory, timing bias, trust triggers).

---

Output in **VALID JSON** only, using this structure:
{{
  "executive_summary": "string",
  "market_positioning": {{
    "uvp": "string",
    "stp_framework": {{
      "segmentation": "string",
      "targeting": "string",
      "positioning": "string"
    }},
    "porter_strategy": "string"
  }},
  "growth_vector": {{
    "selected_paths": ["Market Penetration", "Product Development"],
    "justification": "string"
  }},
  "customer_journey": {{
    "reach": "string",
    "act": "string",
    "convert": "string",
    "engage": "string"
  }},
  "data_driven_plan": {{
    "swot_summary": {{
      "strengths": ["string"],
      "weaknesses": ["string"],
      "opportunities": ["string"],
      "threats": ["string"]
    }},
    "pestle_factors": ["Political", "Economic", "Social", "Technological", "Legal", "Environmental"],
    "smart_goals": ["Specific", "Measurable", "Achievable", "Relevant", "Time-bound"],
    "adaptive_mechanisms": ["Feedback Loops", "A/B Testing", "Real-time Data Reallocation"]
  }},
  "psychological_triggers": [
    "use scarcity to drive urgency",
    "leverage social proof through influencer credibility",
    "create emotional resonance with storytelling"
  ],
  "channels": ["SEO", "Social Media", "Email Marketing", "Paid Ads", "Community Building"],
  "kpis": ["Engagement Rate", "Customer Retention", "Conversion Rate", "Brand Trust Index"]
}}
    """

    return dual_chain_generate(
        prompt=prompt,
        context_type="human_ai_marketing_strategy",
        temperature=temperature,
        use_cache=use_cache,
        web_context=web_context
    )
