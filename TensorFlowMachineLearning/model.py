import os # 警告を消す
os.environ[ 'TF_CPP_MIN_LOG_LEVEL' ] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class featuresAndAnswers:
    def __init__(self, features, answers):
        self.features = features
        self.answers = answers

class trainingAndTest():
    def __init__(self, features, answers, low, high):
        # low - highをテスト, それ以外をトレーニングに使う
        low_number = int(len(features)*low)
        high_number = int(len(features)*high)
        drop_list = range(low_number, high_number)
        test_features = features[low_number:high_number]
        test_answers = answers[low_number:high_number]
        train_features=features.drop(drop_list)
        train_answers=answers.drop(drop_list)
        self.traning = featuresAndAnswers(train_features, train_answers)
        self.test = featuresAndAnswers(test_features, test_answers)
        # 入力及び出力のニューロン数
        self.feature_type_count = len(features.columns)
        self.answer_type_count = len(answers.columns)

class Model:
    def __init__(self, features, answers, layers=[], low=0.8, high=1.0, seeds=1):
        # low - highをテストに使う
        self.data = trainingAndTest(features, answers, low, high)
        # placeholderは変数みたいなもん
        self.real_answer = tf.placeholder(tf.float32,  shape=(None, self.data.answer_type_count))
        self.feature = tf.placeholder(tf.float32, shape=(None, self.data.feature_type_count))
        self.keep_probin = tf.placeholder_with_default(1.0, shape=None)
        self.keep_probhid = tf.placeholder_with_default(1.0, shape=None)
        self.model = self.createTfModel(layers, seeds)
        np.random.seed(seeds)
        # 目標値との誤差 reduce_meanで平均を取る
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.real_answer, logits=self.model))
        # 最適化のアルゴリズム。アダムは評価が高いらしいほかにも10個位tensorflow api にある
        self.step = cost, tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        # 正答率の算出 いつもおんなじ？
        correct_prediction = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.real_answer, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def createTfModel(self, layers, seeds):
        hidden_layer = None
        loop_count_max = len(layers)+1
        for loop_count in range(0,loop_count_max):
            if loop_count is len(layers) :
                # 最後は与えられた答えの数 positive negativeの2になる
                need_answer_count = self.data.answer_type_count
            else :
                # 隠れ層がある場合はその層のニューロンの数？の答えをだすのね
                need_answer_count = layers[loop_count]
            if loop_count is 0 :
                # 最初は与えられた特徴なのね
                feature = self.feature
                feature_type_count = self.data.feature_type_count
                # ドロップアウト
                feature = tf.nn.dropout(feature, self.keep_probin, seed=seeds)
            else :
                # 隠れ層がある場合は隠れ層が学習する特徴のデータになるんだ！
                feature = hidden_layer
                feature_type_count = layers[loop_count-1]
                # ドロップアウト
                feature = tf.nn.dropout(feature, self.keep_probhid, seed=seeds)
            # truncated_normal Tensorを正規分布かつ標準偏差stddevの２倍までのランダムな値で初期化する
            stddev = pow(6/(feature_type_count+need_answer_count), 0.5)
            weights = tf.Variable(tf.truncated_normal([feature_type_count, need_answer_count], stddev=stddev, seed=seeds))
            # バイアス
            biases = tf.Variable(tf.truncated_normal([need_answer_count], stddev=1, seed=seeds))
            # matmulは掛け算feature * weights
            logits = tf.matmul(feature, weights) + biases
            if loop_count is not loop_count_max - 1  :
                # reluはRectified Linear Unit, Rectifier, 正規化線形関数だそうです。
                hidden_layer = tf.nn.relu(logits)
            else :
                # 最後はsoftmax
                return tf.nn.softmax(logits)

    def train(self,count=30000,print_count=10):
        # train_list
        feed_dict = {
            self.feature: self.data.traning.features,
            self.real_answer: self.data.traning.answers,
            self.keep_probin: 0.8,
            self.keep_probhid: 0.5
        }
        # test_list
        feed_dict_t = {
            self.feature: self.data.test.features,
            self.real_answer: self.data.test.answers
        }
        loss_vec = []
        train_acc = []
        test_acc = []
        print("Epoch \t loss")
        for i in range(1, count+1):
            # feed_dictからself.stepを評価, コスト関数を計算
            temp_loss = self.session.run(self.step,feed_dict)
            loss_vec.append(temp_loss[0])
            if i % 10 == 0:
                # trainの正解率, testの正解率を計算
                temp_acc_train = self.session.run(self.accuracy,feed_dict)
                temp_acc_test = self.session.run(self.accuracy,feed_dict_t)
                train_acc.append(temp_acc_train)
                test_acc.append(temp_acc_test)
            if i % (count/print_count) == 0:
                temp = int(count/print_count/4)
                # コスト関数の表示
                print( i, "\t", np.mean(loss_vec[-temp:-1]))
        # Plot loss over time
        plt.plot(loss_vec, 'k-')
        plt.title('Cross Entropy Loss per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Cross Entropy Loss')
        plt.show()
        # Plot train and test accuracy
        plt.plot(train_acc, 'k-', label='Train Set Accuracy')
        plt.plot(test_acc, 'r-', label='Test Set Accuracy')
        plt.title('Train and Test Accuracy')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()

    def test(self):
        # test_list
        feed_dict_t = {
            self.feature: self.data.test.features,
            self.real_answer: self.data.test.answers
        }
        answer = np.round(self.session.run(self.model,feed_dict_t), 0)
        accuracy = []
        for i in range(0, len(self.data.test.answers[0])):
            indexes = [j for j, x in enumerate(self.data.test.answers[:,i]) if x == 1]
            accuracy.append(np.mean(answer[indexes,i]))
        print("Individual accuracy", accuracy)
        accuracy = np.array(answer == self.data.test.answers)
        print ("accuracy", np.mean(accuracy))

    def value(self, ANSWER, LH=[]):
        if len(LH) == 0: number = np.ones(len(self.data.test.features))
        else : number = LH[self.data.test.answers.index]
        answer = self.model.predict(np.array(self.data.test.features))
        data = np.array(self.data.test.features[ANSWER])
        value = [1]
        for i in range(1, len(answer)):
            temp_value = 0
            if number[i-1] >= 0:
                if answer[i-1][0] > 0.5: temp_value = -data[i]
                if answer[i-1][1] > 0.5: temp_value = +data[i]
            value.append(temp_value+value[-1])
        plt.plot(value, 'k-', label='Asset volatility')
        plt.title('Asset volatility')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend(loc='lower right')
        plt.show()
        print("Final_Answer :", answer[-1])
        print("Final_Asset :", value[-1], "\n")