import os

# Đường dẫn thư mục
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

# Đảm bảo thư mục weights tồn tại
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Đường dẫn dữ liệu
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(DATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")
VOCAB_PATH = os.path.join(WEIGHTS_DIR, "vocab.json")

# Hyperparameters tối ưu cho 3050 Ti (4GB VRAM)
MAX_LEN = 128          
BATCH_SIZE = 128       # Tăng batch size để ổn định gradient
D_MODEL = 256          
N_HEAD = 8
NUM_ENCODER_LAYERS = 3 
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1          # Bật lại dropout để chống overfitting

# Hyperparameters của Quá trình Huấn luyện
LEARNING_RATE = 1e-4   
NUM_EPOCHS = 50        # Tăng epoch để đạt mục tiêu CER
PATIENCE = 10          
