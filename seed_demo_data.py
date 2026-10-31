"""
PIXORA AI — Seed Demo Data Script
Generates beautiful sample images with version histories for initial demo evaluation.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import io

from config import Config
from modules.database import init_db, get_platform_metrics
from modules.image_storage import save_uploaded_image, save_edited_version
from modules.database import update_image_caption


def generate_gradient_image(width=1024, height=768, start_color=(20, 25, 45), end_color=(80, 50, 120)) -> Image.Image:
    """Generate a clean gradient image."""
    base = Image.new("RGB", (width, height), start_color)
    draw = ImageDraw.Draw(base)
    for y in range(height):
        factor = y / height
        r = int(start_color[0] * (1 - factor) + end_color[0] * factor)
        g = int(start_color[1] * (1 - factor) + end_color[1] * factor)
        b = int(start_color[2] * (1 - factor) + end_color[2] * factor)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    return base


def create_demo_landscape() -> bytes:
    img = generate_gradient_image(1024, 768, (15, 23, 42), (99, 102, 241))
    draw = ImageDraw.Draw(img)
    # Draw stylized sun/moon
    draw.ellipse([(700, 100), (860, 260)], fill=(255, 230, 150))
    # Draw mountains
    draw.polygon([(0, 768), (300, 380), (600, 768)], fill=(30, 41, 59))
    draw.polygon([(400, 768), (750, 320), (1024, 768)], fill=(15, 23, 42))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_demo_product() -> bytes:
    img = generate_gradient_image(800, 800, (240, 240, 245), (200, 205, 220))
    draw = ImageDraw.Draw(img)
    # Draw sleek modern product bottle / device
    draw.rounded_rectangle([(320, 250), (480, 650)], radius=24, fill=(30, 35, 45), outline=(99, 102, 241), width=4)
    draw.rectangle([(370, 180), (430, 250)], fill=(70, 75, 90))
    # Reflection
    draw.ellipse([(280, 640), (520, 690)], fill=(180, 185, 200))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def seed_demo_library():
    """Seeds database and storage with 2 sample items if empty."""
    init_db()
    metrics = get_platform_metrics()
    if metrics["total_images"] > 0:
        print(f"Library already contains {metrics['total_images']} image(s). Skipping seed.")
        return

    print("Seeding demo data for PIXORA AI...")

    # 1. Landscape demo
    landscape_bytes = create_demo_landscape()
    img1 = save_uploaded_image("aurora_mountain_landscape.png", landscape_bytes)
    update_image_caption(
        img1["id"],
        "A dramatic twilight mountain landscape featuring sharp peaks silhouetted against a vibrant glowing violet-indigo sky with a luminous celestial sun.",
    )

    # Add version 2 to demonstrate version tracking
    v1_pil = Image.open(io.BytesIO(landscape_bytes))
    v2_pil = v1_pil.filter(ImageFilter.GaussianBlur(radius=3))
    draw2 = ImageDraw.Draw(v2_pil)
    draw2.text((50, 50), "Cinematic Atmospheric Filter", fill=(255, 255, 255))
    save_edited_version(
        image_id=img1["id"],
        pil_image=v2_pil,
        prompt="Make the mountain landscape look cinematic with soft twilight atmospheric diffusion",
        provider="Gemini",
        model="imagen-3.0-capability-001",
        parent_version_id=img1.get("active_version_id"),
    )

    # 2. Product demo
    product_bytes = create_demo_product()
    img2 = save_uploaded_image("luxury_perfume_bottle.png", product_bytes)
    update_image_caption(
        img2["id"],
        "A minimalist luxury cosmetic product bottle standing upright on a soft reflective studio backdrop with clean rim lighting and balanced specular highlights.",
    )

    print("Demo data successfully created with versions!")


if __name__ == "__main__":
    seed_demo_library()
