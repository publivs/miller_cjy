import requests
from PIL import Image
from io import BytesIO
from fpdf import FPDF
import os

# 1、用urllib3 或者requests下载jpg
def download_images(image_urls, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images = []
    for idx, url in enumerate(image_urls):
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img_path = os.path.join(save_dir, f'image_{idx}.jpg')
            img.save(img_path)
            images.append(img_path)
        else:
            print(f"Failed to download image from {url}")
    return images

# 排序函数
def sort_strings_by_number(strings):
    import re
    return sorted(strings, key=lambda s: int(re.search(r'\d+', s).group()) if re.search(r'\d+', s) else float('inf'))


def images_to_pdf(image_paths, pdf_path):
    pdf = FPDF()
    image_folder = image_paths
    image_paths = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    # 重新排序
    image_paths = sort_strings_by_number(image_paths)
    for image_path in image_paths:
        jpg_path = os.path.join(image_folder,image_path)
        cover = Image.open(jpg_path)
        width, height = cover.size

        # Convert pixel dimensions to mm with 1px=0.264583 mmi
        width, height = float(width * 0.264583), float(height * 0.264583)
        
        pdf.add_page()
        pdf.image(jpg_path, 0, 0, width, height)

    pdf.output(pdf_path, "F")
def main():
    # 示例图片URL列表
    image_urls=[]
    bese_url="https://doc-w.u.ccb.com/libstorage/convertimages/315d73e1-aeae-4049-bc59-2e9d5c1f54e7/"

    for i in range(0,348):
        image_url = f"{bese_url}{i}.jpg"
        image_urls.append(image_url)

        # 添加更多的图片URL
    
    save_dir = r"C:\Users\cjy\OneDrive\桌面\miller\images"
    pdf_path = "./output_images.pdf"

    # 下载图片
    # image_paths = download_images(image_urls, save_dir)
    
    # 转换为PDF
    images_to_pdf(save_dir, pdf_path)

    print(f"PDF已保存到 {pdf_path}")

if __name__ == "__main__":
    main()


# 2、再用格式工具将jpg转换成pdf,
# 3、保存到指定路径，并完成。
# '''
