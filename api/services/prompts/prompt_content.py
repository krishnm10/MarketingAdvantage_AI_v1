# ==========================================
# ✨ prompt_content.py — Goal-Intent Adaptive Creative Engine (CaaS + RAG)
# ==========================================
import random
import textwrap
import json
from api.services.llm_connector import dual_chain_generate
from api.services.web_search import fetch_relevant_info
from api.services.rag_service import retrieve_relevant_chunks, ingest_content


def safe_parse_json(text: str):
    """Safely parse model output into a JSON object (handles escaped or partial JSON)."""
    if not text:
        return {}
    if isinstance(text, dict):
        return text

    try:
        # Try direct parse
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            # Try unescaping and re-parsing
            text = text.strip("` \n")
            text = text.replace("```json", "").replace("```", "")
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                cleaned = text[start:end + 1]
                return json.loads(cleaned)
        except Exception:
            pass

    # Final fallback
    return {"caption": text.strip()}


def generate_content_prompt(
    business_name: str,
    industry: str,
    goal: str,
    content_type: str,
    mood: str = None,
    tone: str = None,
    temperature: float = 0.85,
    use_cache: bool = True,
    enable_web_search: bool = True,
    enable_rag: bool = True,
    auto_ingest: bool = True
):
    """
    🚀 Goal-Intent Adaptive Content-as-a-Service (CaaS) Engine (RAG + LLM)

    Produces unique, human-aligned creative content with deep understanding of user goals.
    Always returns clean JSON (no escaped strings).
    """

    # 🎲 Creative seed for uniqueness
    variation_seed = random.randint(1000, 9999)
    content_type = content_type.lower().replace(" ", "_")
    mood = mood or random.choice(["inspiring", "educational", "playful", "motivational", "emotional"])
    tone = tone or random.choice(["friendly", "professional", "visionary", "relatable", "empowering"])

    # 🧠 Adaptive Goal Directives for uniqueness
    goal_directives = [
        "Focus on the emotional transformation this goal brings.",
        "Highlight lifestyle, aspiration, and social connection.",
        "Tell a story that makes the goal feel personal and impactful.",
        "Present it as part of a movement or shared experience.",
        "Make it persuasive through human motivation and relatability.",
        "Frame the message around empowerment and self-discovery.",
        "Add an innovative or visionary twist for modern relevance."
    ]
    goal_directive = random.choice(goal_directives)

    # 🧩 Retrieve contextual RAG knowledge
    rag_context = ""
    if enable_rag:
        try:
            retrieved_docs = retrieve_relevant_chunks(f"{industry} {goal}")
            if retrieved_docs:
                joined = "\n\n".join(retrieved_docs)
                rag_context = textwrap.shorten(joined, width=1800, placeholder=" ...[truncated]")
        except Exception as e:
            print(f"[RAG] Retrieval error: {e}")

    # 🌐 Fetch web trends
    web_context = ""
    if enable_web_search:
        try:
            query = f"{industry} {goal} {content_type} creative content trends 2025"
            web_context = fetch_relevant_info(query)
            if web_context:
                web_context = textwrap.shorten(web_context, width=1000, placeholder=" ...[truncated]")
        except Exception as e:
            print(f"[WebSearch] Error fetching context: {e}")

    # 🧠 Intelligent Prompt Construction
    prompt = f"""
(Version Seed #{variation_seed})

You are an elite creative content generator with deep goal-intent reasoning.

🏢 Business: {business_name}
🏭 Industry: {industry}
🎯 Goal: {goal}
🧭 Directive: {goal_directive}

Interpret the user's goal psychologically, emotionally, and contextually.
Understand *why* the goal matters — not just *what* it is.

📚 Verified RAG Knowledge:
{rag_context if rag_context else "N/A"}

🌐 Web Insights:
{web_context if web_context else "N/A"}

📝 Content Type: {content_type}
🎭 Tone: {tone}
💫 Mood: {mood}

Follow rules strictly:
- ❌ No frameworks or marketing strategies.
- ✅ Focus on creativity, emotion, relatability, and clarity.
- ✅ The output must be pure JSON (no explanations, no commentary).
- ✅ Do NOT include markdown, backticks, or text outside JSON.

Output example:
{{
  "headline": "string",
  "caption": "string",
  "hashtags": ["#tag1", "#tag2", "#tag3"],
  "cta": "string",
  "ai_visual_brief": {{
    "style": "string",
    "color_palette": "string",
    "image_prompt": "string",
    "video_prompt": "string"
  }},
  "platform_recommendations": {{
    "instagram": "string",
    "linkedin": "string",
    "twitter": "string",
    "tiktok": "string"
  }},
  "creative_theme": "string",
  "tone_description": "string"
}}
Return only valid JSON.
"""

    # 🚀 LLM Generation
    llm_response = dual_chain_generate(
        prompt=prompt,
        context_type="creative_content_generation",
        temperature=temperature,
        use_cache=use_cache,
        web_context=web_context
    )

    # 🧩 Parse JSON response safely
    content = llm_response.get("content") if isinstance(llm_response, dict) else llm_response
    parsed_content = safe_parse_json(content)

    # 🧠 Optional: Auto-ingest content into RAG
    try:
        if auto_ingest and isinstance(parsed_content, dict):
            combined_text = f"{parsed_content.get('headline', '')}\n{parsed_content.get('caption', '')}".strip()
            if combined_text:
                ingest_content(
                    doc_id=f"{business_name}_{content_type}_{variation_seed}",
                    text=combined_text,
                    metadata={
                        "business_name": business_name,
                        "industry": industry,
                        "goal": goal,
                        "type": content_type,
                        "source": "auto_generated",
                        "mood": mood,
                        "tone": tone,
                        "variation_seed": variation_seed,
                        "goal_directive": goal_directive
                    }
                )
    except Exception as e:
        print(f"[RAG] Auto-ingest failed: {e}")

    # ✅ Return clean structured object
    return {
        "status": "success",
        "message": f"Creative {content_type} generated successfully for {business_name}.",
        "content": parsed_content
    }
