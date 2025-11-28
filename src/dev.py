from models import QNetModelV3

if __name__ == "__main__":
    model = QNetModelV3(n_actions=6, initialization="kaiming", stack_frames=1)
    model.visualize()
