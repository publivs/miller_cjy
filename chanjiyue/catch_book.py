from asyncio import exceptions
import json
import os
import re
from playwright.sync_api import sync_playwright
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from fpdf import FPDF


def save_cache(url, filename='cookies.json'):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(url)
        page.wait_for_timeout(5000)
        input('登录完成后按回车保存 cookies...')
        cookies = page.context.cookies()
        with open(filename, 'w') as f:
            json.dump(cookies, f)
        print("cookies 已保存到", filename)


def load_cache(filename='cookies.json'):
    try:
        with open(filename, 'r') as f:
            cookies = json.load(f)
        print("成功加载 cookies")
        return cookies
    except FileNotFoundError:
        print("未找到 cookies 文件")
        return None


def run(book_url):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.set_viewport_size({"width": 1920, "height": 1080})

        cookies = load_cache()
        if cookies:
            page.context.add_cookies(cookies)

        page.goto('https://u.ccb.com/portal/#/library?microPageId=1685260972055666689')
        page.wait_for_timeout(3000)
        page.goto(book_url)
        page.wait_for_timeout(3000)

        total_page = int(input(' 注意，请输入页数: '))
        try:
            wait_show_mode = input(' 请进入演示模式，确认请按回车 ')
        except Exception as e:
            print(e)
        os.makedirs('images', exist_ok=True)
        for i in range(total_page):
            print(f"抓取第 {i+1}/{total_page} 页")
            page.mouse.click(1900, 980)
            page.wait_for_timeout(2000)
            page.screenshot(path=f'images/{i+1}.png', full_page=True, type='png')
            page.wait_for_timeout(1000)

        browser.close()


def process_images(images_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    png_list = [f for f in os.listdir(images_dir) if f.lower().endswith('.png')]

    for image_name in png_list:
        image_path = os.path.join(images_dir, image_name)
        output_path = os.path.join(output_dir, f"processed_{image_name}")

        img = Image.open(image_path).convert("RGBA")
        arr = np.array(img)

        # 将近黑色像素（R,G,B 均 < 10）变为透明，用 numpy 向量化代替逐像素循环，速度大幅提升
        black_mask = (arr[:, :, 0] < 10) & (arr[:, :, 1] < 10) & (arr[:, :, 2] < 10)
        arr[black_mask] = [0, 0, 0, 0]
        img = Image.fromarray(arr, 'RGBA')

        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

        img = ImageEnhance.Contrast(img).enhance(1.5)
        img = img.filter(ImageFilter.SHARPEN)
        img.save(output_path)
        print(f"处理后的图像已保存为 {output_path}")


def sort_by_number(strings):
    return sorted(strings, key=lambda s: int(m.group()) if (m := re.search(r'\d+', s)) else float('inf'))


def images_to_pdf(image_folder, pdf_path, photo_type='png'):
    pdf = FPDF()
    image_files = sort_by_number([f for f in os.listdir(image_folder) if f.endswith(f'.{photo_type}')])
    for image_name in image_files:
        jpg_path = os.path.join(image_folder, image_name)
        cover = Image.open(jpg_path)
        width, height = cover.size
        pdf.add_page()
        pdf.image(jpg_path, 0, 0, width, height)
    pdf.output(pdf_path, "F")


def pngs_to_pdf(folder_path, output_pdf):
    png_files = sort_by_number([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])
    if not png_files:
        print("未找到 PNG 文件")
        return
    images = [Image.open(f).convert('RGB') for f in png_files]
    images[0].save(output_pdf, save_all=True, append_images=images[1:], quality=95)
    print(f"PDF 已保存到 {output_pdf}")


if __name__ == "__main__":
    # save_cache("https://u.ccb.com/sys/#/login")

    # book_url = 'https://u.ccb.com/lib/reader/reader.html?id=1222c28d-bc43-421a-a452-367b83d89471&pid=02c0dc16-4076-4fdb-a0e4-1e27a333a252'
    # run(book_url)

    process_images('images', 'output_dir')
    pngs_to_pdf('output_dir', 'output_book.pdf')

    # 全部完成后自动关机（Windows），取消注释即可启用
    # os.system("shutdown /s /t 0")
