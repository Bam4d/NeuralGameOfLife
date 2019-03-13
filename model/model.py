import numpy as np
import torch
from torch.nn import BCELoss

from model import StateUpdateAutoEncoder

class GameOfLifeAutoEncoder():

    def __init__(self, width, height):
        self._width = width
        self._height = height

        self._model = StateUpdateAutoEncoder(100, width, height)
        self._criterion = BCELoss()

        self._model.reset_weights()

    def _generate_observation_sequences(self, training_data):

        # Shave off the last element
        observations = training_data[:, :-1].reshape(-1, 2, self._width, self._height)

        # Shave off the first element
        next_observations = training_data[:, 1:].reshape(-1, 2, self._width, self._height)

        # Now our observations and next_observations line up
        return observations, next_observations

    def train(self, epochs, batch_size, learning_rate, training_data_file):

        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate, weight_decay=1e-5)

        training_data = np.load(training_data_file)
        observations, next_observations = self._generate_observation_sequences(training_data)

        total_samples = observations.shape[0]

        training_idx = np.random.rand(len(total_samples)) < 0.8

        training_observations = observations[training_idx]
        training_next_observations = next_observations[training_idx]
        training_samples = training_next_observations.shape[0]

        test_observations = observations[~training_idx]
        test_next_observations = next_observations[~training_idx]
        test_samples = test_next_observations.shape[0]

        for e in range(epochs):


            ###########
            ## TRAIN ##
            ###########
            self._model.train()

            training_batch_loss_values = []

            for batch_start in range(0, training_samples, batch_size):

                # Create the batch
                batch_end = min(batch_start + batch_size, training_samples)

                training_observations_batch = torch.FloatTensor(training_observations[batch_start:batch_end])
                training_next_observations_batch = torch.FloatTensor(training_next_observations[batch_start:batch_end])

                predicted_next_observations_batch = self._model.forward(training_observations_batch)

                loss = self._criterion(predicted_next_observations_batch, training_next_observations_batch)

                # Update the weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                training_batch_loss_values.append(loss.data[0])

            training_loss = np.mean(training_batch_loss_values)

            ##########
            ## TEST ##
            ##########
            self._model.eval()

            test_batch_loss_values = []
            test_prediction_accuracy_values = []

            for batch_start in range(0, test_samples, batch_size):

                # Create the batch
                batch_end = min(batch_start + batch_size, test_samples)

                test_observations_batch = torch.FloatTensor(test_observations[batch_start:batch_end])
                test_next_observations_batch = torch.FloatTensor(test_next_observations[batch_start:batch_end])

                predicted_next_observations_batch = self._model.forward(test_observations_batch)

                loss = self._criterion(predicted_next_observations_batch, test_next_observations_batch)
                loss.backward()

                test_batch_loss_values.append(loss.data[0])

                ##################
                ## TEST ACCURCY ##
                ##################

                # Calculate the accuracy of the predicted observations
                np_next_observation_batch = test_next_observations_batch.cpu().detach().numpy()
                np_next_predicted_observation_batch = predicted_next_observations_batch.cpu().detach().numpy()

                rounded_predicted_next_observation_batch = np.rint(np_next_predicted_observation_batch)

                # Compare the predicted observations with real data
                correct_predictions = 0
                for prediction, target in zip(rounded_predicted_next_observation_batch, np_next_observation_batch):
                    correct_predictions += np.array_equal(prediction, target)

                test_prediction_accuracy_values.append(correct_predictions / batch_size)

            # Average the batch loss and the batch prediction accuracy
            test_loss = np.mean(test_batch_loss_values)
            test_accuracy = np.mean(test_prediction_accuracy_values)

            # Print stuff to output
            print(
                'Epoch [{}/{}], Training loss:{:.4f}, Test loss: {:.4f} - o: {:.4f}'
                    .format(
                    e + 1,
                    epochs,
                    training_loss,
                    test_loss,
                    test_accuracy,
                    )
            )
    # Save the model




    def load_trained(self, trained_model_file):
        trained_model_state_dict = torch.load(trained_model_file, map_location='cpu')
        self._model.load_state_dict(trained_model_state_dict)

    def predict(self, observation):
        torch_observation = torch.FloatTensor(np.expand_dims(observation, 0))
        torch_prediction = self._model.forward(torch_observation)
        return torch_prediction.detach().numpy()