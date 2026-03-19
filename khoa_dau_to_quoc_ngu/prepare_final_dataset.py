import os
import pandas as pd
import glob
from sklearn.model_selection import train_test_split

def prepare_dataset():
    processed_dir = "khoa_dau_to_quoc_ngu/data/processed"
    output_dir = "khoa_dau_to_quoc_ngu/data"
    
    # 1. Thu thập tất cả các file đã xử lý
    all_files = glob.glob(os.path.join(processed_dir, "*_processed.csv"))
    if not all_files:
        print(f"Lỗi: Không tìm thấy file đã xử lý nào tại {processed_dir}")
        return

    print(f"Đang gộp {len(all_files)} tệp dữ liệu...")
    
    # 2. Đọc và gộp dữ liệu
    df_list = []
    for f in all_files:
        df_list.append(pd.read_csv(f))
    
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Loại bỏ trùng lặp và các câu quá ngắn (nếu có)
    initial_len = len(full_df)
    full_df = full_df.drop_duplicates().dropna()
    print(f"Tổng số câu sau khi gộp và lọc trùng: {len(full_df):,} (Đã loại bỏ {initial_len - len(full_df):,} câu trùng)")

    # 3. Chia dữ liệu: 90% Train, 5% Val, 5% Test
    print("Đang chia tập dữ liệu (90/5/5)...")
    train_df, temp_df = train_test_split(full_df, test_size=0.1, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # 4. Lưu kết quả
    print(f"Đang lưu các tập dữ liệu vào {output_dir}...")
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False, encoding='utf-8')
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False, encoding='utf-8')
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False, encoding='utf-8')

    print("-" * 30)
    print(f"HOÀN TẤT CHUẨN BỊ DATASET:")
    print(f" - Tập Huấn luyện (Train): {len(train_df):,} câu")
    print(f" - Tập Kiểm thử (Val):     {len(val_df):,} câu")
    print(f" - Tập Đánh giá (Test):    {len(test_df):,} câu")
    print("-" * 30)

if __name__ == "__main__":
    prepare_dataset()
