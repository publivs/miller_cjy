import json
import os
from playwright.sync_api import sync_playwright
import time
from lxml import etree
from PIL import Image, ImageEnhance, ImageFilter
from fpdf import FPDF


def save_cache(url, filename='cookies.json'):
    """进行登录并询问用户是否保存 cookies 到 JSON 文件"""
    # 填写登录表单（根据实际情况修改选择器）

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context()

        page = context.new_page()
        page.goto(url)
        # 等待页面加载
        page.wait_for_timeout(5000)  # 等待 5 秒钟以查看页面

        # 询问用户是否保存 cookies
        # user_input = input("是否保存当前 cookies? (y/n): ")
        input('----------y/n-----------')
        cookies = page.context.cookies()
        with open(filename, 'w') as f:
            json.dump(cookies, f)
        print("cookies 已保存到", filename)


def load_cache(filename='cookies.json'):
    """从 JSON 文件加载 cookies"""
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
        # 尝试加载之前保存的 cookies
        cookies = load_cache()
        time.sleep(3)
        if cookies:
            page.context.add_cookies(cookies)

        page.goto('https://u.ccb.com/portal/#/library?microPageId=1685260972055666689')
        time.sleep(3)
        # 访问目标页面
        page.goto(book_url)

        point_root = page.content()
        F = etree.HTML(point_root)

        # 在这里可以添加其他操作，例如截图或数据提取
        # page.wait_for_timeout(8000)  # 等待 5 秒钟以查看页面

        total_page =  int(input(' ---------------------------- 注意,请输入页数 ---------------------------- '))
        input(' ---------------------------- 请进入演示模式,确认请按Y ---------------------------- ')

        for i in range(total_page):
            print(f"抓取第{i+1}页")
            page.mouse.click(1900,980)
            time.sleep(2)
            page.screenshot(path=f'images/{i+1}.png', full_page=True,type='png')
            time.sleep(1)

        # 关闭浏览器
        browser.close()



def process_images(images_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有图像文件
    png_list = os.listdir(images_dir)
    for image_name in png_list:
        image_path = os.path.join(images_dir, image_name)
        output_path = os.path.join(output_dir, f"processed_{image_name}")

        # 打开图像
        img = Image.open(image_path)

        # 转换为 RGBA（支持透明度）
        img = img.convert("RGBA")

        # 获取图像数据
        data = img.getdata()

        # 创建一个新的列表，用于存储非黑色像素
        new_data = []
        for item in data:
            # 只保留非黑色像素（RGB=(0, 0, 0)）
            if item[0] < 10 and item[1] < 10 and item[2] < 10:  # 允许小范围的黑色
                new_data.append((0, 0, 0, 0))  # 将黑色像素变为透明
            else:
                new_data.append(item)

        # 更新图像数据
        img.putdata(new_data)

        # 裁剪图像
        img = img.crop(img.getbbox())

        # 调整对比度
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)  # 1.5 为对比度系数，值越高对比度越强

        # 应用锐化
        img = img.filter(ImageFilter.SHARPEN)

        # 保存处理后的图像
        img.save(output_path)
        print(f"处理后的图像已保存为 {output_path}")

def sort_strings_by_number(strings):
    import re
    return sorted(strings, key=lambda s: int(re.search(r'\d+', s).group()) if re.search(r'\d+', s) else float('inf'))


def images_to_pdf(image_paths, pdf_path,photo_type = 'png'):
    pdf = FPDF()
    image_folder = image_paths
    image_paths = [f for f in os.listdir(image_folder) if f.endswith(f'.{photo_type}')][:8]
    # 重新排序
    image_paths = sort_strings_by_number(image_paths)
    for image_path in image_paths:
        print(image_path)
        jpg_path = os.path.join(image_folder,image_path)
        cover = Image.open(jpg_path)
        width, height = cover.size

        # Convert pixel dimensions to mm with 1px=0.264583 mmi
        # width, height = float(width * 0.264583), float(height * 0.264583)
        pdf.add_page()
        pdf.image(jpg_path, 0, 0, width, height)

    pdf.output(pdf_path, "F")

def pngs_to_pdf(folder_path, output_pdf):
    # 获取文件夹中所有的PNG文件
    png_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]

    # 确保PNG文件按名称排序
    png_files = sort_strings_by_number(png_files)

    # 打开所有PNG文件并转换为PDF
    images = [Image.open(png_file).convert('RGB') for png_file in png_files]

    # 保存为高质量PDF
    images[0].save(output_pdf, save_all=True, append_images=images[1:], quality=95)


if __name__ == "__main__":
    # save_cache("https://u.ccb.com/sys/#/login")

    # book_url = 'https://u.ccb.com/lib/reader/reader.html?id=f906b8d9-93f6-4d14-bf8a-1ed22f056ec8'
    # run(book_url)

    # process_images('images','output_dir')

    pngs_to_pdf('output_dir', 'output_book.pdf')