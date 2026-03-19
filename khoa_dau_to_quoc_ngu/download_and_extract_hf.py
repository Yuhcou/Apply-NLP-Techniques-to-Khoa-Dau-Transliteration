import os
from datasets import load_dataset
from tqdm import tqdm

def download_and_save_locally(dataset_name, output_dir):
    """Tải toàn bộ dataset từ Hugging Face và lưu thành các file cục bộ."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Đang tải dataset {dataset_name} (Lưu ý: Dung lượng khoảng 24GB)...")
    # Tải dataset (sẽ lưu vào cache của Hugging Face trên máy bạn)
    dataset = load_dataset(dataset_name, split='train')
    
    total_rows = len(dataset)
    shard_size = 1000000  # Mỗi file cục bộ chứa 1 triệu dòng
    num_shards = (total_rows // shard_size) + 1
    
    print(f"Tổng cộng {total_rows} dòng. Bắt đầu chia nhỏ thành {num_shards} shards...")
    
    for i in range(num_shards):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, total_rows)
        
        if start_idx >= total_rows:
            break
            
        shard_file = os.path.join(output_dir, f"corpus_shard_{i:03d}.parquet")
        
        # Nếu shard đã tồn tại thì bỏ qua (hỗ trợ resume)
        if os.path.exists(shard_file):
            continue
            
        print(f"Đang lưu shard {i+1}/{num_shards} ({start_idx} -> {end_idx})...")
        # Chỉ lấy cột 'text' để tối ưu dung lượng
        shard = dataset.select(range(start_idx, end_idx))
        shard.to_parquet(shard_file)

    print(f"--- HOÀN TẤT ---")
    print(f"Toàn bộ dữ liệu đã được lưu tại: {output_dir}")

if __name__ == "__main__":
    DATASET_NAME = "BlossomsAI/vietnamese-corpus"
    # Lưu vào thư mục data của project
    LOCAL_RAW_DIR = "khoa_dau_to_quoc_ngu/data/raw_corpus"
    
    download_and_save_locally(DATASET_NAME, LOCAL_RAW_DIR)
