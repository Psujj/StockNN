import PythonScripts.NN as NN
import PythonScripts.Dataset as DS
import PythonScripts.RandomNumber as Rd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

newModel = False
train = False
test = True

device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model is running on: {device}")

if newModel:
    model = NN.StockPredictor().to(device)
else: 
    model = NN.StockPredictor().to(device)
    #model.load_state_dict(torch.load('stock_model.pth'))
    model.load_state_dict(torch.load('smallModelHighCost.pth'))
    
    model.eval()  # Switch to inference mode

#for graph
losses = []
direction_accuracies = []
up_prediction_percentages = []
#for graph

trainingFeatures, testNormalized = DS.create_dataset()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
minIndex = 14
ntraining = 2299

if train:
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct_direction = 0
        total_valid = 0
        predicted_up_count = 0
        predicted_down_count = 0
        # actual_change_count = 0

        random_Indexes = Rd.randomIndex(ntraining - minIndex,epoch*12,minIndex) # want to geenrate random indexes such that can access 14 days
        for index in random_Indexes:
            InputArray, ExpectedCloseOutput = DS.createInput(trainingFeatures, index)
            if InputArray is None: # invalid index, must skip
                continue
            tensorInput = torch.tensor(InputArray, dtype=torch.float32).reshape(1, -1).to(device)
            ExpectedCloseOutput = torch.tensor(ExpectedCloseOutput, dtype=torch.float32).to(device)

            output = model(tensorInput)
            loss = model.loss(output, ExpectedCloseOutput, tensorInput)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            total_loss += loss.item()
            prev_close = tensorInput[-1][0].item()  # last clos
            actual_change = ExpectedCloseOutput.item() - prev_close
            predicted_change = output.item() - prev_close

            # print(f"Prev Close: {prev_close:.4f} | Expected: {ExpectedCloseOutput.item()} | Predicted: {predicted_change:.4f} | ΔActual: {actual_change}")

            if actual_change * predicted_change > 0:  # correct direction
                correct_direction += 1
            total_valid += 1

            if predicted_change > 0:
                predicted_up_count += 1
            else:
                predicted_down_count += 1

        #     if actual_change > 0:
        #         actual_change_count += 1
        # averageChange = actual_change_count/total_valid

        averageLoss = total_loss / len(random_Indexes)
        averageDirection = correct_direction/total_valid
        averageUpPredictionPercentage = predicted_up_count / (predicted_up_count + predicted_down_count)
        losses.append(averageLoss)
        direction_accuracies.append(averageDirection)
        up_prediction_percentages.append(averageUpPredictionPercentage)
        

        print(f"Epoch {epoch + 1}, Loss: {averageLoss}, Correct Direction : {averageDirection * 100}, UP Prediction %: {averageUpPredictionPercentage * 100}")
            #, Values: {averageChange}

if test: 
    model.eval()  # switch to eval mode
    test_loss = 0.0
    correct_test_direction = 0
    test_valid = 0
    test_up_count = 0
    test_down_count = 0

    with torch.no_grad():  # no need to track gradients during testing
        for index in range(minIndex, len(testNormalized)):
            testInput, testExpectedOutput = DS.createInput(testNormalized, index)
            if testInput is None:
                continue

            tensorTestInput = torch.tensor(testInput, dtype=torch.float32).reshape(1, -1).to(device)
            testExpectedOutput = torch.tensor(testExpectedOutput, dtype=torch.float32).to(device)

            testOutput = model(tensorTestInput)
            test_loss += model.loss(testOutput, testExpectedOutput, tensorTestInput).item()

            prev_close = tensorTestInput[-1][0].item()
            actual_change = testExpectedOutput.item() - prev_close
            predicted_change = testOutput.item() - prev_close

            if actual_change * predicted_change > 0:
                correct_test_direction += 1
            test_valid += 1

            if predicted_change > 0:
                test_up_count += 1
            else:
                test_down_count += 1

    # Compute averages
    average_test_loss = test_loss / test_valid
    average_test_accuracy = correct_test_direction / test_valid
    average_up_pred_pct = test_up_count / (test_up_count + test_down_count)

    print("\nTest Set Evaluation")
    print(f"Test Loss: {average_test_loss}")
    print(f"Test Direction Accuracy: {average_test_accuracy * 100}%")
    print(f"Test UP Prediction %: {average_up_pred_pct * 100}%")
    print(len(testNormalized))

#save trained model: 
torch.save(model.state_dict(), 'stock_model.pth')



# create and save graph
plt.figure(figsize=(12, 5))

# Loss graph
plt.subplot(1, 2, 1)
plt.plot(losses[:1000], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.xscale('log')
plt.legend()

# Direction accuracy and UP prediction graph
plt.subplot(1, 2, 2)
plt.plot(direction_accuracies[:1000], color='orange', label='Direction Accuracy')
plt.plot(up_prediction_percentages[:1000], color='green', linestyle='--', label='UP Prediction %')
plt.xlabel('Epoch (log scale)')
plt.ylabel('Percentage')
plt.title('Direction Accuracy and UP Prediction %')
plt.xscale('log')
plt.legend()

plt.tight_layout()
plt.savefig("training_metrics.png", dpi=300)
