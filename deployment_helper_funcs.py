from model import load_saved_model, predict

model = load_saved_model(load_last=True)

def predict_rt(img):
    return predict(model, img)

