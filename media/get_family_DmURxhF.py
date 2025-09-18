import vt
import asyncio
import os
import csv

# 配置参数
API_KEY = "7bb0497a13057a4cf8f03a249bf8f0ea09ce15b6f456f598e1cf988c3c1732f6" # 替换为你的API密钥
MD5_LIST_FILE = r"md5_mapping_final.txt" # 更新为正确路径
OUTPUT_FILE = "results.csv" # 结果文件保存到与代码同级目录

def get_md5_list(filename):
    """读取MD5列表文件，格式为 md5,index"""
    try:
        with open(filename, 'r') as f:
            return [line.strip().split(',') for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误: 文件 '{filename}' 未找到，请检查文件路径是否正确。")
        raise
    except Exception as e:
        print(f"读取文件 '{filename}' 时发生错误: {e}")
        raise

async def process_md5(client, md5, index):
    """处理单个MD5查询并返回结果"""
    try:
        file_obj = await client.get_object_async(f"/files/{md5}")
        file_obj_dict = file_obj.to_dict()
        
        # 提取家族信息
        family = []
        if 'crowdsourced_ids' in file_obj_dict.get('attributes', {}):
            for entry in file_obj_dict['attributes']['crowdsourced_ids']:
                if isinstance(entry, dict) and 'value' in entry and isinstance(entry['value'], str):
                    family.append(entry['value'])
        
        if not family and 'last_analysis_results' in file_obj_dict.get('attributes', {}):
            vendors = list(file_obj_dict['attributes']['last_analysis_results'].keys())[:3]
            family = [file_obj_dict['attributes']['last_analysis_results'][vendor]['result'] for vendor in vendors
                      if file_obj_dict['attributes']['last_analysis_results'][vendor].get('result')]
        
        if not family and 'popular_threat_classification' in file_obj_dict.get('attributes', {}):
            threat_info = file_obj_dict['attributes']['popular_threat_classification']
            if 'popular_threat_name' in threat_info:
                family = [item['value'] for item in threat_info['popular_threat_name'] if isinstance(item, dict) and 'value' in item]
        
        family = list(set([f.strip() for f in family if f.strip()]))
        
        # 提取类别
        category = "N/A"
        if 'popular_threat_classification' in file_obj_dict.get('attributes', {}):
            threat_info = file_obj_dict['attributes']['popular_threat_classification']
            if 'popular_threat_category' in threat_info:
                category = [item['value'] for item in threat_info['popular_threat_category'] if isinstance(item, dict) and 'value' in item]
                category = ", ".join(category) if category else "N/A"
            elif 'suggested_threat_label' in threat_info:
                category = threat_info['suggested_threat_label'].split('.')[0]
        
        return {
            "md5": md5,
            "index": index,
            "category": category,
            "family": ", ".join(family) if family else "N/A"
        }
    except vt.error.APIError as e:
        if e.code == "NotFoundError":
            print(f"[!] {md5} (index: {index}) 未找到")
        else:
            print(f"[!] {md5} (index: {index}) 查询失败: {e}")
        return None
    except Exception as e:
        print(f"[!] {md5} (index: {index}) 处理时发生错误: {e}")
        return None

async def main():
    client = None
    try:
        # 初始化客户端
        client = vt.Client(API_KEY)
        
        # 读取MD5列表
        md5_list = get_md5_list(MD5_LIST_FILE)
        
        # 初始化输出文件
        if not os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["md5", "index", "category", "family"])
                writer.writeheader()
        
        # 逐个处理MD5
        for idx, (md5, index) in enumerate(md5_list):
            print(f"Processing {idx+1}/{len(md5_list)}: {md5} (index: {index})")
            
            # 获取结果
            result = await process_md5(client, md5, index)
            
            # 写入结果
            if result:
                with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=["md5", "index", "category", "family"])
                    writer.writerow(result)
                    print(f"已写入: {md5} (index: {index})")
            
            # 遵守API速率限制（免费版4次/分钟）
            await asyncio.sleep(15)
    
    except Exception as e:
        print(f"主程序发生错误: {e}")
    finally:
        # 确保客户端关闭
        if client:
            await client.close_async()
            print("客户端已关闭")

if __name__ == "__main__":
    asyncio.run(main())