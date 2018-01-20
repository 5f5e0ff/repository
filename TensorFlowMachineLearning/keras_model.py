import os  # 警告を消す
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Activation, normalization, Dropout
from keras.initializers import RandomNormal, glorot_normal
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class featuresAndAnswers:
    def __init__(self, features, answers):
        self.features = features
        self.answers = answers
        # 入力及び出力のニューロン数
        self.feature_type_count = len(features.columns)
        self.answer_type_count = len(answers.columns)


class trainingAndTest():
    def __init__(self, features, answers, low, high):
        # low - highをテスト, それ以外をトレーニングに使う
        low_number = int(len(features) * low)
        high_number = int(len(features) * high)
        drop_list = range(low_number, high_number)
        test_features = features[low_number:high_number]
        test_answers = answers[low_number:high_number]
        train_features = features.drop(drop_list)
        train_answers = answers.drop(drop_list)
        self.traning = featuresAndAnswers(train_features, train_answers)
        self.test = featuresAndAnswers(test_features, test_answers)


class Model:
    def __init__(self, features, answers, layers=[], seeds=1):
        # 乱数シード
        np.random.seed(seeds)
        tf.set_random_seed(seeds)
        config = tf.ConfigProto(
                intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
            )
        sess = tf.Session(graph=tf.get_default_graph(), config=config)
        K.set_session(sess)
        # データの作成
        self.rawdata = featuresAndAnswers(features, answers)
        # モデルの生成
        self.model = self.createTfModel(layers)
        # 初期重みの保存
        self.model.save_weights('ini_weights.hdf5')
        # モデルのサマリの確認
        self.model.summary()

    def createTfModel(self, layers):
        model = Sequential()
        # 入力層
        input_n = self.rawdata.feature_type_count
        model.add(InputLayer(input_shape=(input_n, )))
        model.add(normalization.BatchNormalization())
        model.add(Dropout(0.8))
        # 中間層
        for hidden_n in layers:
            model.add(Dense(hidden_n,
                kernel_initializer=glorot_normal(),
                bias_initializer=RandomNormal(stddev=1)
            ))
            #model.add(normalization.BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
        # 出力層
        output_n = self.rawdata.answer_type_count
        model.add(Dense(output_n,
            kernel_initializer=glorot_normal(),
            bias_initializer=RandomNormal(stddev=1)
        ))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
            optimizer='adam',metrics=['accuracy']
        )
        return model

    def initial(self, low=0.8, high=1.0):
        # 初期重みを読み込み
        self.model.load_weights('ini_weights.hdf5')
        # テストデータの作成
        self.data = trainingAndTest(
                self.rawdata.features,
                self.rawdata.answers, low, high
            )
        print("\n――――― testdata : {:04d} - {:04d} ――――――――――".format(
            int(len(self.rawdata.features) * low),
            int(len(self.rawdata.features) * high - 1)
        ))

    def train(self, epochs=1000, batch_size=32, patience=64):
        # トレーニング, テストデータを代入
        train_x = np.array(self.data.traning.features)
        train_y = np.array(self.data.traning.answers)
        test_x = np.array(self.data.test.features)
        test_y = np.array(self.data.test.answers)
        # 学習
        mc_cb = ModelCheckpoint(filepath='best_weights.hdf5',monitor='val_acc',
                save_best_only=True,save_weights_only=True
            )
        es_cb = EarlyStopping(monitor='val_loss', patience=patience)
        history = self.model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,
                verbose=0,callbacks=[mc_cb, es_cb],validation_data=(test_x, test_y)
            )
        self.model.load_weights('best_weights.hdf5')
        # 損失の履歴をプロット
        plt.plot(history.history['loss'], label="loss")
        plt.plot(history.history['val_loss'], label="val_loss")
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='lower right')
        plt.show()
        # 精度の履歴をプロット
        plt.plot(history.history['acc'], label="accuracy")
        plt.plot(history.history['val_acc'], label="val_acc")
        plt.title('model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(loc="lower right")
        plt.show()

    def test(self):
        # テストデータを代入
        test_x = np.array(self.data.test.features)
        test_y = np.array(self.data.test.answers)
        answer = self.model.predict(test_x)
        for i in range(0, len(answer)):
            answer[i] = answer[i] / max(answer[i])
        answer = answer.astype(int)
        accuracy = []
        for i in range(0, len(test_y[0])):
            indexes = [j for j, x in enumerate(test_y[:, i]) if x == 1]
            accuracy.append(np.mean(answer[indexes, i]))
        # 各要素の正解率
        print("Individual accuracy :", accuracy)
        accuracy = np.array(answer == test_y)
        # 全体の正解率
        print("accuracy :", np.mean(accuracy))

    def value(self, ANSWER, DAY, LH=[]):
        if len(LH) == 0: number = np.ones(len(self.data.test.features))
        else: number = LH[self.data.test.answers.index]
        answer = self.model.predict(np.array(self.data.test.features))
        data = np.array(self.data.test.features[ANSWER + "_" + str(DAY)])
        value = [1]
        kaka = DAY
        kuku = 0.5
        aa = [0, 0]
        for i in range(kaka, len(answer)):
            temp_value = 0
            if number[i - kaka] >= 0:
                if answer[i - kaka][0] > kuku: temp_value = -data[i]
                if answer[i - kaka][1] > kuku: temp_value = +data[i]
            value.append(temp_value + value[-1])
            if temp_value > 0: aa[1] += 1
            if temp_value < 0: aa[0] += 1
        plt.plot(value, 'k-', label='Asset volatility')
        plt.title('Asset volatility')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend(loc='lower right')
        plt.show()
        print("Final_Answer :", answer[-1])
        print("accuracy :", aa[1] / (aa[0] + aa[1]))
        print("Final_Asset :", value[-1])