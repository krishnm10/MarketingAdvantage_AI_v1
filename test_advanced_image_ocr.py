"""
test_advanced_image_ocr.py — Test the enhanced image OCR pipeline

Tests:
  1. Advanced multi-pass OCR with OpenCV pre-processing
  2. Color region text extraction
  3. VisualExplainer semantic interpretation
  4. Full captioner → explainer pipeline

Usage:
  python test_advanced_image_ocr.py <image_path>
  python test_advanced_image_ocr.py <image_path> --full-pipeline

If no image path is provided, creates a synthetic bar chart for testing.
"""

import asyncio
import sys
import os
import textwrap

# ── Ensure project root is on sys.path ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_test_chart(output_path: str = "test_chart_generated.png"):
    """Generate a synthetic bar chart image for testing OCR extraction."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("ERROR: Pillow required. Install with: pip install Pillow")
        return None

    width, height = 800, 500
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((250, 20), "India Population Growth", fill="black")
    draw.text((300, 40), "(in Crore)", fill="gray")

    # Y-axis labels
    for i, val in enumerate([0, 40, 80, 120, 160]):
        y = 400 - (i * 85)
        draw.text((30, y - 8), str(val), fill="black")
        draw.line([(70, y), (750, y)], fill="#e0e0e0", width=1)

    # Bars with data
    bar_data = [
        ("1951", 36, "#e74c3c"),   # Red
        ("1971", 54, "#3498db"),   # Blue
        ("1991", 84, "#2ecc71"),   # Green
        ("2001", 102, "#f39c12"),  # Orange
        ("2011", 121, "#9b59b6"),  # Purple
        ("2021", 138, "#1abc9c"),  # Teal
    ]

    bar_width = 80
    spacing = 110
    start_x = 100

    for i, (year, value, color) in enumerate(bar_data):
        x = start_x + i * spacing
        bar_height = int((value / 160) * 340)
        y_top = 400 - bar_height

        # Draw colored bar
        draw.rectangle([x, y_top, x + bar_width, 400], fill=color)

        # Draw value on top of bar
        draw.text((x + 25, y_top - 20), str(value), fill="black")

        # Draw year label below
        draw.text((x + 20, 410), year, fill="black")

    # Growth rate labels
    draw.text((180, 440), "Growth Rate: 21.5%  24.8%  23.9%  21.5%  17.7%  --", fill="#555555")
    draw.text((250, 465), "Source: Census of India", fill="gray")

    img.save(output_path)
    print(f"✅ Test chart saved to: {output_path}")
    return output_path


async def test_captioner(image_path: str):
    """Test the enhanced ImageCaptionerCPU."""
    from app.ai.providers.local.image_caption_cpu_v1 import ImageCaptionerCPU

    print("\n" + "=" * 70)
    print("  ADVANCED IMAGE CAPTIONER TEST")
    print("=" * 70)
    print(f"  Image: {image_path}")
    print("=" * 70)

    captioner = ImageCaptionerCPU()
    result = await captioner.caption(image_path)

    print(f"\n📋 Caption: {result.get('caption', 'N/A')}")
    print(f"📊 Is Chart: {result.get('is_chart', False)}")

    ocr_text = result.get("ocr_text", "")
    print(f"\n📝 OCR Text ({len(ocr_text)} chars):")
    print("-" * 50)
    for line in ocr_text.splitlines():
        if line.strip():
            print(f"  {line}")
    print("-" * 50)

    objects = result.get("objects")
    if objects:
        print(f"\n🔢 Extracted Data Values ({len(objects)}):")
        for val in objects:
            print(f"  • {val}")

    return result


async def test_visual_explainer(ocr_text: str):
    """Test the VisualExplainerCPU semantic interpretation."""
    from app.ai.providers.local.visual_explainer_cpu_v1 import VisualExplainerCPU

    print("\n" + "=" * 70)
    print("  VISUAL EXPLAINER TEST")
    print("=" * 70)

    explainer = VisualExplainerCPU()
    explanation = await explainer.explain(ocr_text)

    print(f"\n🧠 Semantic Explanation ({len(explanation)} chars):")
    print("-" * 50)
    for line in explanation.splitlines():
        print(f"  {line}")
    print("-" * 50)

    return explanation


async def test_full_pipeline(image_path: str):
    """Test captioner → explainer → synthesis pipeline."""
    from app.ai.providers.local.image_caption_cpu_v1 import ImageCaptionerCPU
    from app.ai.providers.local.visual_explainer_cpu_v1 import VisualExplainerCPU

    print("\n" + "=" * 70)
    print("  FULL PIPELINE TEST (Captioner → Explainer → Synthesis)")
    print("=" * 70)

    # Step 1: Caption
    captioner = ImageCaptionerCPU()
    result = await captioner.caption(image_path)
    caption = result.get("caption", "")
    ocr_text = result.get("ocr_text", "")
    is_chart = result.get("is_chart", False)
    data_values = result.get("objects", [])

    print(f"\n  Step 1 — Caption: {caption}")
    print(f"           Is Chart: {is_chart}")
    print(f"           OCR Lines: {len(ocr_text.splitlines())}")
    print(f"           Data Values: {len(data_values or [])}")

    # Step 2: Explain (if chart-like)
    explanation = ""
    if is_chart and ocr_text:
        explainer = VisualExplainerCPU()
        explanation = await explainer.explain(ocr_text)
        print(f"\n  Step 2 — Visual Explanation:")
        for line in explanation.splitlines():
            print(f"           {line}")
    else:
        print(f"\n  Step 2 — Skipped (not a chart or no OCR text)")

    # Step 3: Final synthesis (mirrors production _synthesize_text_with_explainer)
    from app.services.ingestion.media.image_ingestor_v1 import ImageIngestorV1
    clean_ocr = ImageIngestorV1._clean_ocr_for_output(ocr_text) if ocr_text else ""

    print(f"\n  Step 3 — Final Semantic Output:")
    print("-" * 50)
    if explanation:
        output = f"{caption}\n\n--- Semantic Analysis ---\n{explanation}"
        if clean_ocr:
            output += f"\n\n--- Raw OCR ---\n{clean_ocr}"
    else:
        output = f"{caption}\n\nDetected text:\n{clean_ocr}" if clean_ocr else caption

    for line in output.splitlines():
        print(f"  {line}")
    print("-" * 50)
    print(f"\n  Total output: {len(output)} chars")

    return output


async def main():
    image_path = None
    full_pipeline = "--full-pipeline" in sys.argv

    # Get image path from args
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            image_path = arg
            break

    # If no image provided, generate a test chart
    if not image_path:
        print("No image path provided — generating a synthetic test chart...")
        image_path = create_test_chart()
        if not image_path:
            print("Failed to generate test chart.")
            return

    if not os.path.exists(image_path):
        print(f"ERROR: File not found: {image_path}")
        return

    # Run tests
    result = await test_captioner(image_path)

    ocr_text = result.get("ocr_text", "")
    if ocr_text:
        await test_visual_explainer(ocr_text)

    if full_pipeline:
        await test_full_pipeline(image_path)

    print("\n✅ All tests complete.")


if __name__ == "__main__":
    asyncio.run(main())
