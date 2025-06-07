import os

ONNX_EXPORT_PATTERN_V1 = "python tools/export_onnx.py -c configs/rtdetrv2/rtdetrv2_r%dvd_6x_coco.yml -r scheckpoint/rtdetrv2_r%dvd_6x_coco_ema.pth --check"
RUNNING_CMD_PATTERN_V1 = "python tools/train.py -c configs/rtdetrv2/rtdetrv2_r%dvd_6x_coco.yml -r scheckpoint/rtdetrv2_r%dvd_6x_coco_ema.pth --test-only"
SIZE_LIST = [18, 34, 50, 101]
LOG_ROOT = "logs"
MODEL_ROOT = "models"
def get_eval_log_v1_sz(suffix="a", sz=50, n = 10):
    LOG_DIR = "%s/eval_%s" % (LOG_ROOT, suffix)
    os.makedirs(LOG_DIR, exist_ok=True)
    for i in range(n):
        out_path_i = "%s/eval_%d_%d.txt" % (LOG_DIR, sz,i)
        cmd = RUNNING_CMD_PATTERN_V1 % (sz,sz)
        cmd += " > " + out_path_i
        print("Running: ", cmd)
        os.system(cmd)
def engine_export(suffix="a", sz=50):
    # onnx:
    cmd = ONNX_EXPORT_PATTERN_V1 % (sz, sz)
    print("Running: ", cmd)
    os.system(cmd)
    cmd = "python tools/export_trt.py -i model.onnx"
    print(cmd)
    os.system(cmd)
    os.makedirs(MODEL_ROOT, exist_ok=True)
    cmd = "mv model.onnx %s/model_%d_%s.onnx" % (MODEL_ROOT, sz, suffix)
    os.system(cmd)
    cmd = "mv model.engine %s/model_%d_%s.engine" % (MODEL_ROOT, sz, suffix)
    os.system(cmd)
def engine_export_all(suffix="a"):
    for sz in SIZE_LIST:
        engine_export(suffix,sz)

def get_eval_log_v1_all(suffix="a", n = 10):
    for sz in SIZE_LIST:
        get_eval_log_v1_sz(suffix=suffix, sz=sz, n = n)

if __name__ == "__main__":
    # get_eval_log_v1_all(suffix="a",n=5)
    engine_export_all(suffix="a")