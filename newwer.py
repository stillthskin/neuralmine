# Modelclass
class Model:
    def __init__(self):
        # Createalistofnetworkobjects
        self.layers = []
        # Softmaxclassifier'soutputobject
        self.softmax_classifier_output = None
    # Addobjectstothemodel

    def add(self, layer):
        self.layers.append(layer)
    # Setloss,optimizerandaccuracy

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    # Finalizethemodel

    def finalize(self):
        # Createandsettheinputlayer
        self.input_layer = Layer_Input()
        # Countalltheobjects
        layer_count = len(self.layers)
        # Initializealistcontainingtrainablelayers:
        self.trainable_layers = []
        # Iteratetheobjects
        for i in range(layer_count):
            # If it's the first layer,
            # #thepreviouslayerobjectistheinputlayer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
                # Alllayersexceptforthefirstandthelast
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            # Thelastlayer-thenextobjectistheloss
            # Alsolet'ssaveasidethereferencetothelastobject
            # whoseoutputisthemodel'soutput
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            # Iflayercontainsanattributecalled"weights",
            # it'satrainablelayer-
            # addittothelistoftrainablelayers
            # Wedon'tneedtocheckforbiases-
            # checkingforweightsisenough
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
                # Updatelossobjectwithtrainablelayers
                self.loss.remember_trainable_layers(self.trainable_layers)
            # IfoutputactivationisSoftmaxand
            # lossfunctionisCategoricalCross-Entropy
            # createanobjectofcombinedactivation
            # andlossfunctioncontaining
            # fastergradientcalculation
            if isinstance(self.layers[-1], Activation_Softmax) and \
                    isinstance(self.loss, Loss_CategoricalCrossentropy):
                # Createanobjectofcombinedactivation
                # andlossfunctions
                self.softmax_classifier_output =\
                    Activation_Softmax_Loss_CategoricalCrossentropy()
    # Trainthemodel

    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        # Initializeaccuracyobject
        self.accuracy.init(y)
        # Defaultvalueifbatchsizeisnotbeingset
        train_steps = 1
        # Ifthereisvalidationdatapassed,
        # setdefaultnumberofstepsforvalidationaswell
        if validation_data is not None:
            validation_steps = 1
            # Forbetterreadability
            X_val, y_val = validation_data
        # Calculatenumberofsteps
        if batch_size is not None:
            train_steps = len(X)//batch_size
            # Dividingroundsdown.Iftherearesomeremaining
            # databutnotafullbatch,thiswon'tincludeit
            # Add`1`toincludethisnotfullbatch
        if train_steps * batch_size < len(X):
            train_steps += 1
        if validation_data is not None:
            validation_steps = len(X_val)//batch_size
            # Dividingroundsdown.Iftherearesomeremaining
            # databutnorfullbatch,thiswon'tincludeit
            # Add`1`toincludethisnotfullbatch
        if validation_steps * batch_size < len(X_val):
            validation_steps += 1
    # Maintrainingloop
    for epoch in range(1, epochs+1):
        # Printepochnumber
        print(f'epoch:{epoch}')
        # Resetaccumulatedvaluesinlossandaccuracyobjects
        self.loss.new_pass()
        self.accuracy.new_pass()
        # Iterateoversteps
        for step in range(train_steps):
            # Ifbatchsizeisnotset-
            # trainusingonestepandfulldataset
            if batch_size is None:
                batch_X = X
                batch_y = y
            # Otherwisesliceabatch
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]
                batch_y = y[step*batch_size:(step+1)*batch_size]
            # Performtheforwardpass
            output = self.forward(batch_X, training=True)
            # Calculateloss
            data_loss, regularization_loss =\
                self.loss.calculate(output, batch_y,
                                    include_regularization=True)
            loss = data_loss+regularization_loss
            # Getpredictionsandcalculateanaccuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, batch_y)
            # Performbackwardpass
            self.backward(output, batch_y)
            # Optimize(updateparameters)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()
            # Printasummary
            if not step % print_every or step == train_steps - 1:
                print(f'step:{step},' +
                      f'acc:{accuracy:.3f},' +
                      f'loss:{loss:.3f}(' +
                      f'data_loss:{data_loss:.3f},' +
                      f'reg_loss:{regularization_loss:.3f}),' +
                      f'lr:{self.optimizer.current_learning_rate}')
        # Getandprintepochlossandaccuracy
        epoch_data_loss, epoch_regularization_loss = \
            self.loss.calculate_accumulated(
                include_regularization=True)
        epoch_loss = epoch_data_loss+epoch_regularization_loss
        epoch_accuracy = self.accuracy.calculate_accumulated()
        print(f'training,' +
              f'acc:{epoch_accuracy:.3f},' +
              f'loss:{epoch_loss:.3f}(' +
              f'data_loss:{epoch_data_loss:.3f},' +
              f'reg_loss:{epoch_regularization_loss:.3f}),' +
              f'lr:{self.optimizer.current_learning_rate}')
        # Ifthereisthevalidationdata
        if validation_data is not None:
            # Resetaccumulatedvaluesinloss
            # #andaccuracyobjects
            self.loss.new_pass()
            self.accuracy.new_pass()
            # Iterateoversteps
            for step in range(validation_steps):
                # Ifbatchsizeisnotset-
                # trainusingonestepandfulldataset
                if batch_size is None:
                    batch_X = X_val
                    batch_y = y_val
                # Otherwisesliceabatch
                else:
                    batch_X = X_val[step*batch_size:(step+1)*batch_size]
                    batch_y = y_val[step*batch_size:(step+1)*batch_size]
                # Performtheforwardpass
                output = self.forward(batch_X, training=False)
                # Calculatetheloss
                self.loss.calculate(output, batch_y)
            # Getpredictionsandcalculateanaccuracy
            predictions = self.output_layer_activation.predictions(
                output)
            self.accuracy.calculate(predictions, batch_y)
            # Getandprintvalidationlossandaccuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        # Printasummary
        print(f'validation,' +
              f'acc:{validation_accuracy:.3f},' +
              f'loss:{validation_loss:.3f}')
    # Performsforwardpass

    def forward(self, X, training):
        # Callforwardmethodontheinputlayer
        # thiswillsettheoutputpropertythat
        # thefirstlayerin"prev"objectisexpecting
        self.input_layer.forward(X, training)
        # Callforwardmethodofeveryobjectinachain
        # Passoutputofthepreviousobjectasaparameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        # "layer"isnowthelastobjectfromthelist,
        # returnitsoutput
        return layer.output
    # Performsbackwardpass

    def backward(self, output, y):
        # Ifsoftmaxclassifier
        if self.softmax_classifier_output is not None:
            # Firstcallbackwardmethod
            # onthecombinedactivation/loss
            # thiswillsetdinputsproperty
            self.softmax_classifier_output.backward(output, y)
            # Sincewe'llnotcallbackwardmethodofthelastlayer
            # whichisSoftmaxactivation
            # asweusedcombinedactivation/loss
            # object,let'ssetdinputsinthisobject
            self.layers[-1].dinputs =\
                self.softmax_classifier_output.dinputs
            # Callbackwardmethodgoingthrough
            # alltheobjectsbutlast
            # inreversedorderpassingdinputsasaparameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
                return
        # Firstcallbackwardmethodontheloss
        # thiswillsetdinputspropertythatthelast
        # layerwilltrytoaccessshortly
        self.loss.backward(output, y)
        # Callbackwardmethodgoingthroughalltheobjects
        # inreversedorderpassingdinputsasaparameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
# Loads a MNIST dataset


def load_mnist_dataset(dataset, path):
    # Scanallthedirectoriesandcreatealistoflabels
    labels = os.listdir(os.path.join(path, dataset))
    # Createlistsforsamplesandlabels
    X = []
    y = []
    # Foreachlabelfolder
    for label in labels:
        # Andforeachimageingivenfolder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Readtheimage
            image = cv2.imread(
                os.path.join(path, dataset, label, file),
                cv2.IMREAD_UNCHANGED)
            # Andappenditandalabeltothelists
            X.append(image)
            y.append(label)
    # Convertthedatatopropernumpyarraysandreturn
    return np.array(X), np.array(y).astype('uint8')
# MNISTdataset(train+test)


def create_data_mnist(path):
    # Loadbothsetsseparately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    # Andreturnallthedata
    return X, y, X_test, y_test


# Createdataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')
# Shufflethetrainingdataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]
# Scaleandreshapesamples
X = (X.reshape(X.shape[0], -1).astype(np.float32)-127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5)/127.5
# Instantiatethemodel
model = Model()
# Addlayers
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())
# Setloss,optimizerandaccuracyobjects
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=1e-4),
    accuracy=Accuracy_Categorical()
)
# Finalizethemodel
model.finalize()
# Trainthemodel
model.train(X, y, validation_data=(X_test, y_test),
            epochs=10, batch_size=128, print_every=100)
