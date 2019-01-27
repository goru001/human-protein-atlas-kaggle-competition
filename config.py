class DefaultConfigs(object):
    train_data = "/home/gaurav/Downloads/data/protein/train/"
    test_data = "/home/gaurav/Downloads/data/protein/test/"
    weights = "./checkpoints/"
    best_models = "./checkpoints/best_models"
    best_models_256 = "./checkpoints/best_models/256_best/"
    best_models_512 = "./checkpoints/best_models/512_best/"
    submit = "./submit/"
    model_name = "bninception_bcelog"
    num_classes = 28
    img_weight = 512
    img_height = 512
    channels = 4
    lr = 0.003
    # lr = 0.001
    batch_size = 32
    epochs = 5
    folds = 7
    fastai_models = "./checkpoints/fastai/"


config = DefaultConfigs()
