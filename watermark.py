from ultralytics import YOLO
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.live import Live
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from vibrant import Vibrant
from typing import List, Dict, Tuple
import argparse

# Constants
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp']
WATERMARK_BOX_WIDTH_RATIO = 2 # 워터마크 가로 길이 = 얼굴 영역의 가로 길이 * 입력값
WATERMARK_BOX_HEIGHT_RATIO = 0.3 # 워터마크 세로 길이 = 얼굴 영역의 세로 길이 * 입력값
WATERMARK_TEXT_SIZE_RATIO = 1.0 # 워터마크 텍스트 크기 = 워터마크 세로 길이 * 입력값
WATERMARK_VERTICAL_OFFSET_RATIO = 0.5 # 워터마크 세로 위치 = 얼굴 위치 아랫부분 + 얼굴 영역의 세로 길이 * 입력값
DEFAULT_BG_COLOR = (128, 128, 128, 192) # 배경 색상 감지 실패시 기본 적용 색상
DEFAULT_TEXT_COLOR = (255, 255, 255) # 텍스트 색상 감지 실패시 기본 적용 색상

console = Console()
logging.getLogger("ultralytics").setLevel(logging.ERROR)

def load_image(img_path: Path) -> Image.Image:
    """
    주어진 경로의 이미지를 로드합니다.

    Args:
        img_path (Path): 로드할 이미지 경로

    Returns:
        Image.Image: 로드된 이미지
    """
    
    return Image.open(img_path)

def split_image(img: Image.Image) -> List[Image.Image]:
    """
    입력 이미지를 여러 개의 정사각형 이미지로 분할합니다.

    Args:
        img (Image.Image): 분할할 원본 이미지

    Returns:
        List[Image.Image]: 분할된 이미지
    """
    
    width, height = img.size
    split_size = width
    splits = []
    for i in range(0, height, split_size):
        split = img.crop((0, i, width, min(i+split_size, height)))
        if split.size[1] < split_size:
            new_split = Image.new(img.mode, (width, split_size), (0, 0, 0))
            new_split.paste(split, (0, 0))
            split = new_split
        splits.append(split)
    return splits

def merge_images(splits: List[Image.Image]) -> Image.Image:
    """
    분할된 이미지들을 하나의 이미지로 병합합니다.

    Args:
        splits (List[Image.Image]): 병합할 이미지 리스트

    Returns:
        Image.Image: 병합된 이미지
    """
    
    return Image.fromarray(np.vstack([np.array(split) for split in splits]))

def extract_colors(face_area: Image.Image) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int]]:
    """
    얼굴 영역에서 배경색과 텍스트 색상을 추출합니다.

    Args:
        face_area (Image.Image): 얼굴 영역 이미지

    Returns:
        Tuple[Tuple[int, int, int, int], Tuple[int, int, int]]: 배경색(RGBA)과 텍스트 색상(RGB)
    """
    
    v = Vibrant()
    try:
        palette = v.get_palette(face_area)
        bg_color = tuple(palette.muted.rgb) + (192,) if palette.muted else DEFAULT_BG_COLOR
        bg_color_2 = tuple(palette.light_muted.rgb) + (192,) if palette.light_muted else DEFAULT_BG_COLOR
        text_color = tuple(int(c * 1.1) if c * 1.1 <= 255 else 255 for c in palette.light_muted.rgb) if palette.light_muted else DEFAULT_TEXT_COLOR
    except Exception as e:
        bg_color = DEFAULT_BG_COLOR
        bg_color_2 = DEFAULT_BG_COLOR
        text_color = DEFAULT_TEXT_COLOR
    return bg_color, bg_color_2, text_color

def draw_watermark(draw: ImageDraw.ImageDraw, split: Image.Image, face: List[float], watermark_text: str, watermark_logo: Image.Image, font_path: str) -> None:
    """
    이미지에 워터마크를 그립니다.

    Args:
        draw (ImageDraw.ImageDraw): PIL ImageDraw
        split (Image.Image): 워터마크를 그릴 이미지 분할
        face (List[float]): 얼굴 영역의 좌표
        watermark_text (str): 워터마크 텍스트
        watermark_logo (Image.Image): 워터마크 로고 이미지
        font_path (str): 폰트 파일 경로

    Returns:
        None
    """
    
    x1, y1, x2, y2 = map(int, face[:4])
    face_width = x2 - x1
    face_height = y2 - y1
    
    watermark_width = int(face_width * WATERMARK_BOX_WIDTH_MULTIPLIER)
    watermark_height = int(face_height * WATERMARK_BOX_HEIGHT_MULTIPLIER)

    y_offset = int(face_height * WATERMARK_VERTICAL_OFFSET_MULTIPLIER)
    watermark_x = (x1+x2-watermark_width)//2
    watermark_y = y2 + y_offset

    color_area = split.crop((x1, y1 + y_offset, x2, y2 + y_offset))
    bg_color, bg_color_2, text_color = extract_colors(color_area)
    
    # Ensure bg_color is in RGBA format
    if len(bg_color) == 3:
        bg_color = bg_color + (255,)  # Add full opacity if it's missing

    if watermark_logo:
        # 로고 워터마크 적용
        logo_width, logo_height = watermark_logo.size
        logo_aspect_ratio = logo_width / logo_height

        # 워터마크 영역의 가로 길이에 맞추어 로고 크기 조정
        new_logo_width = watermark_width
        new_logo_height = int(new_logo_width / logo_aspect_ratio)

        resized_logo = watermark_logo.resize((new_logo_width, new_logo_height), Image.LANCZOS)
        tinted_logo = apply_tint(resized_logo, bg_color, bg_color_2)
        adjusted_logo = adjust_logo_opacity(tinted_logo, opacity=0.8)  # 80% 투명도 적용

        # 로고를 워터마크 영역의 중앙에 배치
        logo_x = watermark_x + (watermark_width - new_logo_width) // 2
        logo_y = watermark_y + (watermark_height - new_logo_height) // 2

        split.paste(adjusted_logo, (logo_x, logo_y), adjusted_logo)
    else:
        # 텍스트 워터마크 적용
        font_size = int(watermark_height * WATERMARK_TEXT_SIZE_MULTIPLIER)
        font = ImageFont.truetype(font_path, font_size)

        text_box = draw.textbbox((0, 0), watermark_text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        
        text_x = watermark_x + (watermark_width - text_width) // 2
        text_y = watermark_y + (watermark_height - text_height) // 2

        watermark_box = Image.new('RGBA', (watermark_width, watermark_height))
        draw_gradient = ImageDraw.Draw(watermark_box)
        for x in range(watermark_width):
            r = int(bg_color[0] + (bg_color_2[0] - bg_color[0]) * x / watermark_width)
            g = int(bg_color[1] + (bg_color_2[1] - bg_color[1]) * x / watermark_width)
            b = int(bg_color[2] + (bg_color_2[2] - bg_color[2]) * x / watermark_width)
            a = int(bg_color[3] + (bg_color_2[3] - bg_color[3]) * x / watermark_width)
            draw_gradient.line([(x, 0), (x, watermark_height)], fill=(r, g, b, a))

        split.paste(watermark_box, (watermark_x, watermark_y), mask=watermark_box)

        draw.text((text_x, text_y -5), watermark_text, font=font, fill=text_color)

def process_split(split: Image.Image, model: YOLO, watermark_text: str, watermark_logo: Image.Image, font_path: str) -> Tuple[Image.Image, int]:
    """
    이미지 분할을 처리하고 워터마크를 추가합니다.

    Args:
        split (Image.Image): 처리할 이미지 분할
        model (YOLO): 얼굴 감지 모델(YOLOv8)
        watermark_text (str): 워터마크 텍스트
        watermark_logo (Image.Image): 워터마크 로고 이미지
        font_path (str): 폰트 파일 경로

    Returns:
        Tuple[Image.Image, int]: 처리된 이미지 분할과 감지된 얼굴 수
    """
    
    rgb_split = split.convert('RGB') # RGB로 변환
    results = model(np.array(rgb_split), verbose=False)
    num_faces = len(results[0].boxes)

    draw = ImageDraw.Draw(split)

    for result in results:
        faces = result.boxes.xyxy
        for face in faces:
            draw_watermark(draw, split, face, watermark_text, watermark_logo, font_path)

    return split, num_faces

def process_image(img_path: Path, output_folder: Path, model: YOLO, watermark_text: str, watermark_logo: Image.Image, font_path: str) -> Dict[str, any]:
    """
    각 이미지에 워터마크를 추가합니다.

    Args:
        img_path (Path): 처리할 이미지 경로
        output_folder (Path): 출력 폴더 경로
        model (YOLO): 얼굴 감지 모델(YOLOv8)
        watermark_text (str): 워터마크 텍스트
        watermark_logo (Image.Image): 워터마크 로고 이미지
        font_path (str): 폰트 파일 경로

    Returns:
        Dict[str, any]: 처리 결과 (이미지 이름, 감지된 얼굴 수)
    """
    
    img = load_image(img_path)
    original_size = img.size
    splits = split_image(img)
    processed_splits = []
    total_faces = 0

    for split in splits:
        processed_split, num_faces = process_split(split, model, watermark_text, watermark_logo, font_path)
        processed_splits.append(processed_split)
        total_faces += num_faces

    processed_img = merge_images(processed_splits)
    processed_img = processed_img.crop((0, 0, original_size[0], original_size[1]))

    output_img_path = output_folder / f"processed_{img_path.name}"
    processed_img.save(str(output_img_path))

    return {
        "image_name": img_path.name,
        "num_faces": total_faces
    }

def process_images(input_folder: str, output_folder: str, model_path: str, watermark_text: str, watermark_logo: str, font_path: str) -> None:
    """
    입력 폴더의 모든 이미지를 처리합니다.

    Args:
        input_folder (str): 입력 이미지 폴더 경로
        output_folder (str): 출력 이미지 폴더 경로
        model_path (str): 얼굴 감지 모델 파일 경로(YOLOv8)
        watermark_text (str): 워터마크 텍스트
        watermark_logo (str): 워터마크 로고 이미지 파일 경로
        font_path (str): 폰트 파일 경로

    Returns:
        None
    """
    
    model = YOLO(model_path)
    logo_image = process_logo(watermark_logo) if watermark_logo else None

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in input_folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn()
    )

    all_results = []

    with Live(progress, refresh_per_second=10):
        task = progress.add_task("[green]처리중...", total=len(image_files))

        for img_path in image_files:
            result = process_image(img_path, output_folder, model, watermark_text, logo_image, font_path)
            all_results.append(result)
            progress.update(task, advance=1, description=f"[green]처리중... ({img_path.name})")

    display_results(all_results)

def display_results(results: List[Dict[str, any]]) -> None:
    """
    처리 결과를 테이블 형식으로 출력합니다.

    Args:
        results (List[Dict[str, any]]): 처리 결과 리스트

    Returns:
        None
    """
    
    table = Table(title="워터마크 삽입 완료")
    table.add_column("파일명", style="cyan")
    table.add_column("워터마크(개)", style="magenta")

    for result in results:
        table.add_row(result["image_name"], str(result["num_faces"]))

    console.print(table)

def process_logo(logo_path: str) -> Image.Image:
    """
    로고 이미지를 처리합니다. 배경을 제거하고 알파 채널을 조정합니다.

    Args:
        logo_path (str): 로고 이미지 파일 경로

    Returns:
        Image.Image: 처리된 로고 이미지
    """
    logo = Image.open(logo_path)
    logo = logo.convert('RGBA')
    
    # 이미지에 이미 알파 채널이 있는지 확인
    if logo.mode == 'RGBA' and any(pixel[3] < 255 for pixel in logo.getdata()):
        # 알파 채널이 있는 경우, 처리하지 않고 그대로 반환
        return logo
    
    # 이미지를 그레이스케일로 변환
    gray = ImageOps.grayscale(logo)
    
    # Otsu's 방법을 사용하여 임계값 설정
    threshold = int(np.mean(gray))
    
    # 알파 채널 생성
    alpha = Image.new('L', logo.size, 0)
    for x in range(logo.width):
        for y in range(logo.height):
            if gray.getpixel((x, y)) <= threshold:
                alpha.putpixel((x, y), 255)
            else:
                alpha.putpixel((x, y), 0)
    
    # 원본 이미지에 알파 채널 적용
    logo.putalpha(alpha)
    
    return logo

def apply_tint(image: Image.Image, bg_color: Tuple[int, int, int, int], bg_color_2: Tuple[int, int, int, int]) -> Image.Image:
    """
    이미지에 그라디언트 틴트를 적용합니다.

    Args:
        image (Image.Image): 원본 이미지
        bg_color (Tuple[int, int, int, int]): 시작 색상 (RGBA)
        bg_color_2 (Tuple[int, int, int, int]): 끝 색상 (RGBA)

    Returns:
        Image.Image: 그라디언트 틴트가 적용된 이미지
    """
    # Ensure the image is in RGBA mode
    image = image.convert('RGBA')
    
    # Create a grayscale version for tinting
    grayscale = image.convert('L')
    
    # Increase brightness by 50%
    brightened = Image.eval(grayscale, lambda x: 255)
    
    width, height = image.size
    tinted_image = Image.new('RGBA', (width, height))
    
    for y in range(height):
        for x in range(width):
            progress = (x / width + y / height) / 2
            r = int(bg_color[0] + (bg_color_2[0] - bg_color[0]) * progress)
            g = int(bg_color[1] + (bg_color_2[1] - bg_color[1]) * progress)
            b = int(bg_color[2] + (bg_color_2[2] - bg_color[2]) * progress)
            a = image.getpixel((x, y))[3]
            
            if a != 0:
                gray = brightened.getpixel((x, y))
                nr = int(gray * r / 255)
                ng = int(gray * g / 255)
                nb = int(gray * b / 255)
                tinted_image.putpixel((x, y), (nr, ng, nb, a))
            else:
                tinted_image.putpixel((x, y), (0, 0, 0, 0))
    
    return tinted_image

def adjust_logo_opacity(image: Image.Image, opacity: float = 0.5) -> Image.Image:
    """
    로고 이미지의 불투명도를 조정합니다.

    Args:
        image (Image.Image): 원본 로고 이미지
        opacity (float): 적용할 불투명도 (0.0 ~ 1.0)

    Returns:
        Image.Image: 불투명도가 조정된 이미지
    """
    assert 0 <= opacity <= 1, "Opacity must be between 0 and 1"
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    alpha = image.split()[3]
    alpha = alpha.point(lambda p: int(p * opacity))
    
    result = image.copy()
    result.putalpha(alpha)
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="워터마크 삽입 도구")
    parser.add_argument("-t", "--text", type=str, default="SAMPLE", help="워터마크 텍스트")
    parser.add_argument("-l", "--logo", type=str, help="워터마크 로고 이미지 파일 경로")
    input_folder = "input"
    output_folder = "output"
    model_path = "model.pt"
    watermark_text = "SAMPLE"
    watermark_text = args.text if not args.logo else None
    watermark_logo = args.logo
    font_path = "font.otf"
    process_images(input_folder, output_folder, model_path, watermark_text, watermark_logo, font_path)