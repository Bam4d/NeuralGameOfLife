from model import GameOfLifeAutoEncoder

if __name__ == '__main__':

    width = 10
    height = 10

    model = GameOfLifeAutoEncoder(width, height)

    model.train(150, 2048, 0.0005,'training_data.npy')

    model.save_trained_model('trained_model.tch')