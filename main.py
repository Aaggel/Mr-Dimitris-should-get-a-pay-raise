# βιβλιοθήκες / libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score

#για αργότερα / for later
best_case1_model = None
best_case2_model = None
best_case1_acc = 0
best_case2_acc = 0
best_case1_cm = None
best_case2_cm = None
results_case1 = []
results_case2 = []

# φτιάχνουμε τυχαία δεδομένα / making some random data
def generate_data(samples):
    ages = np.random.randint(13, 101, samples)
    weights = np.random.randint(40, 121, samples)
    labels = [] # για να κρατάμε κάπου τα νούμερα / to keep the numbers somewhere
    for age, weight in zip(ages, weights):
        prob = 0.0 # πιθανότητες (0 = τίποτα κ 1 = παρενέργειες)  / probabilities (0 = none and 1 = sideffects)
        if age >= 65:
            prob += 0.9 # +90% 
        else:
            prob += 0.1 # -90%
        if weight >= 70:
            prob += 0.05 # -10%
        else:
            prob -= 0.05 # +10%
        prob = max(0, min(1, prob)) 
        label = 1 if np.random.rand() < prob else 0 # για να μην βγαίνουν πάντα τα ίδια / element of surprise!
        labels.append(label)

    x = np.column_stack((ages, weights))
    y = np.array(labels)
    return x, y

# ετοιμάζουμε τα δεδομένα μας / preprocess our data
def preprocess_data(x, y):
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled,
        y,
        test_size=0.2,
        random_state=42
    )
    return x_train, x_test, y_train, y_test, scaler

# για να μην κρασάρει το mse / so mse won't crash
def fix_labels_for_mse(y):
    return np.eye(2)[y]

# Πρώτη περίπτωση / First case
def create_model_case1(learning_rate, loss_function):
    model = Sequential()
    model.add(Input(shape=(2,)))  
    model.add(Dense(16, activation='relu')) # πρώτο κρυμμένο επίπεδο / first hidden layer
    model.add(Dense(32, activation='relu')) # δεύτερο κρυμμένο επίπεδο / second hiden layer
    model.add(Dense(2, activation='softmax')) # δύο έξοδοι / two outputs
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['accuracy']
    )
    return model

# Δεύτερη περίπτωση / Second case
def create_model_case2(learning_rate, loss_function):
    model = Sequential()
    model.add(Input(shape=(2,)))  
    model.add(Dense(16, activation='relu')) # πρώτο κρυμμένο επίπεδο / first hidden layer
    model.add(Dense(32, activation='relu')) # δεύτερο κρυμμένο επίπεδο / second hiden layer
    model.add(Dense(64, activation='relu')) # τρίτο κρυμμένο επίπεδο / third hiden layer
    model.add(Dense(2, activation='softmax')) # ξανά δύο έξοδοι / two outputs again
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['accuracy']
    )
    return model

# ρυθμίσεις (μπορείτε να τα αλλάξετε αυτά!) / settings (you can change these!)
sample_sizes = [1500, 3000]
learning_rates = [0.01, 0.0001]
epochs_list = [20, 30]
batch_sizes = [10, 50]
# δοκιμάζουμε με δύο loss functions / we try with two loss functions
loss_functions = ['sparse_categorical_crossentropy', 'mean_squared_error']

# Εκπαίδευση και αξιολόγηση / training and evaluation
def train_and_evaluate(model, x_train, x_test, y_train, y_test, epochs, batch_size, title, loss_function):
    if loss_function == 'mean_squared_error':
        y_train_use = fix_labels_for_mse(y_train)
        y_test_use = fix_labels_for_mse(y_test)
    else:
        y_train_use = y_train
        y_test_use = y_test
    model.fit(x_train, y_train_use, validation_split=0.2,
              epochs=epochs, batch_size=batch_size, verbose=0)
    # για να τα βλέπει από όλες τις συναρτήσεις / so it can see them through all functions
    global best_case1_model, best_case2_model, best_case1_acc, best_case2_acc, best_case1_cm, best_case2_cm
    # Προβλέψεις / Predictions
    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(y_test, y_pred) # Ακρίβεια / Accuracy
    cm = confusion_matrix(y_test, y_pred) # Confusion matrix
    # για να τυπωθεί ωραία / so it will print nicely :)
    print("\n===ᓚ₍ ^. .^₎===₍ᐢ. .ᐢ₎===")
    print(title)
    print("========≽^- ˕ -^≼========")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(cm)
    print("\n /)/)\n( ˶•༝•)\n୭( づ✿")
    # για να κρατάει τις καλύτερες περιπτώσεις / to keep the best cases
    if title == "CASE1":
        results_case1.append((cm, accuracy))
        if accuracy > best_case1_acc:
            best_case1_acc = accuracy
            best_case1_cm = cm
            best_case1_model = model
    else:
        results_case2.append((cm, accuracy))
        if accuracy > best_case2_acc:
            best_case2_acc = accuracy
            best_case2_cm = cm
            best_case2_model = model

# τα κυρίως / main stuff
for samples in sample_sizes:
    # φτιάχνουμε δεδομένα / we generate some data
    x, y = generate_data(samples)
    # τα καθαρίζουμε / we clean them up
    x_train, x_test, y_train, y_test, scaler = preprocess_data(x, y)
    for lr in learning_rates:
        for loss_fn in loss_functions:
            for epochs in epochs_list:
                for batch_size in batch_sizes:
                    # για την πρώτη περίπτωση! / for first case!
                    model1 = create_model_case1(lr, loss_fn)
                    train_and_evaluate(model1, x_train, x_test, y_train, y_test, epochs, batch_size, "CASE1", loss_fn)
                    # για την δεύτερη περίπτωση! / for the second case!
                    model2 = create_model_case2(lr, loss_fn)
                    train_and_evaluate(model2, x_train, x_test, y_train, y_test, epochs, batch_size, "CASE2", loss_fn)

# Οι καλύτερες περιπτώσεις :) / best cases!
print("\nBEST CASE1 ACC:", best_case1_acc)
print(best_case1_cm)
print("\nBEST CASE2 ACC:", best_case2_acc)
print(best_case2_cm)

# Τεστάρουμε με καινούργιες τιμές / testing with new values
print("\nTest data!")
new_patients = np.array([
    [20, 60],
    [70, 90],
    [70, 60],
    [80, 50],
    [30, 78]
]) # έβαλα και μερικές οριακές για να δούμε τι θα κάνει / I also put some borderline ones in there, to see what it will do
print(new_patients)

new_patients_scaled = scaler.transform(new_patients) #κι εδώ καθαρίζουμε (με τον ίδιο scaler) / we clean up here too (with the same scaler)

print("CASE1 PREDICTIONS") # με δύο επίπεδα / with two layers
preds1 = best_case1_model.predict(new_patients_scaled, verbose=0)
print(np.argmax(preds1, axis=1))

print("CASE2 PREDICTIONS") # με τρία επίπεδα / with three layers
preds2 = best_case2_model.predict(new_patients_scaled, verbose=0)
print(np.argmax(preds2, axis=1))

# να είστε όλοι καλά! / may you all be well! c:
