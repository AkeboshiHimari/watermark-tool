from ultralytics import YOLO
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.live import Live
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from vibrant import Vibrant
from typing import List, Dict, Tuple

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
        text_color = tuple(palette.light_muted.rgb) if palette.light_muted else DEFAULT_TEXT_COLOR
    except Exception as e:
        bg_color = DEFAULT_BG_COLOR
        text_color = DEFAULT_TEXT_COLOR
    return bg_color, text_color

def draw_watermark(draw: ImageDraw.ImageDraw, split: Image.Image, face: List[float], watermark_text: str, font_path: str) -> None:
    """
    이미지에 워터마크를 그립니다.

    Args:
        draw (ImageDraw.ImageDraw): PIL ImageDraw
        split (Image.Image): 워터마크를 그릴 이미지 분할
        box (List[float]): 얼굴 영역의 좌표
        watermark_text (str): 워터마크 텍스트
        font_path (str): 폰트 파일 경로

    Returns:
        None
    """
    
    x1, y1, x2, y2 = map(int, face[:4])
    # 얼굴 영역
    face_width = x2 - x1
    face_height = y2 - y1
    
    # 워터마크 영역
    watermark_width = int(face_width * WATERMARK_BOX_WIDTH_RATIO)
    watermark_height = int(face_height * WATERMARK_BOX_HEIGHT_RATIO)

    # 워터마크 위치
    y_offset = int(face_height * WATERMARK_VERTICAL_OFFSET_RATIO)
    watermark_x = (x1+x2-watermark_width)//2
    watermark_y = y2 + y_offset
    
    # 워터마크 텍스트
    font_size = int(watermark_height * WATERMARK_TEXT_SIZE_RATIO)
    font = ImageFont.truetype(font_path, font_size)

    # 워터마크 텍스트 영역
    text_box = draw.textbbox((0, 0), watermark_text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    
    # 텍스트 위치
    text_x = watermark_x + (watermark_width - text_width) // 2
    text_y = watermark_y + (watermark_height - text_height) // 2

    # 색상 추출
    color_area = split.crop((x1, y1 + y_offset, x2, y2 + y_offset))
    bg_color, text_color = extract_colors(color_area)

    # 워터마크 삽입
    watermark_box = Image.new('RGBA', (watermark_width, watermark_height), bg_color)
    split.paste(watermark_box, (watermark_x, watermark_y), mask=watermark_box)



    draw.text((text_x, text_y -5), watermark_text, font=font, fill=text_color)

def process_split(split: Image.Image, model: YOLO, watermark_text: str, font_path: str) -> Tuple[Image.Image, int]:
    """
    이미지 분할을 처리하고 워터마크를 추가합니다.

    Args:
        split (Image.Image): 처리할 이미지 분할
        model (YOLO): 얼굴 감지 모델(YOLOv8)
        watermark_text (str): 워터마크 텍스트
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
            draw_watermark(draw, split, face, watermark_text, font_path)

    return split, num_faces

def process_image(img_path: Path, output_folder: Path, model: YOLO, watermark_text: str, font_path: str) -> Dict[str, any]:
    """
    각 이미지에 워터마크를 추가합니다.

    Args:
        img_path (Path): 처리할 이미지 경로
        output_folder (Path): 출력 폴더 경로
        model (YOLO): 얼굴 감지 모델(YOLOv8)
        watermark_text (str): 워터마크 텍스트
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
        processed_split, num_faces = process_split(split, model, watermark_text, font_path)
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

def process_images(input_folder: str, output_folder: str, model_path: str, watermark_text: str, font_path: str) -> None:
    """
    입력 폴더의 모든 이미지를 처리합니다.

    Args:
        input_folder (str): 입력 이미지 폴더 경로
        output_folder (str): 출력 이미지 폴더 경로
        model_path (str): 얼굴 감지 모델 파일 경로(YOLOv8)
        watermark_text (str): 워터마크 텍스트
        font_path (str): 폰트 파일 경로

    Returns:
        None
    """
    
    model = YOLO(model_path)

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
            result = process_image(img_path, output_folder, model, watermark_text, font_path)
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

if __name__ == "__main__":
    input_folder = "input"
    output_folder = "output"
    model_path = "model.pt"
    watermark_text = "SAMPLE"
    font_path = "font.otf"
    process_images(input_folder, output_folder, model_path, watermark_text, font_path)