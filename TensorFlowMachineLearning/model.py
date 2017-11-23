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
        low_number = int(len(features)*low)
        high_number = int(len(features)*high)
        test_features = features[low_number:high_number]
        test_answers = answers[low_number:high_number]
        for i in range(low_number, high_number):
            features=features.drop([i])
            answers=answers.drop([i])
        self.traning = featuresAndAnswers(features, answers)
        self.test = featuresAndAnswers(test_features, test_answers)
        # N225_1 とかいっぱい
        self.feature_type_count = len(features.columns)
        # positive nagativeの２つ
        self.answer_type_count = len(answers.columns)

class Model:
    def __init__(self, features, answers, layers=[], low=0.8, high=1.0, seeds=1):
        # low-highをトレーニングに使う
        self.data = trainingAndTest(features, answers, low, high)
        # placeholderは変数みたいなもん
        self.real_answer = tf.placeholder(tf.float32,  shape=(None, self.data.answer_type_count))
        self.feature = tf.placeholder(tf.float32, shape=(None, self.data.feature_type_count))
        self.model = self.createTfModel(layers, seeds)
        # 目標値との誤差 reduce_meanで平均を取る 1e-10~1.0に正規化することでlog0を回避
        cost = -tf.reduce_mean(self.real_answer*tf.log(tf.clip_by_value(self.model,1e-10,1.0)))
        # 最適化のアルゴリズム。アダムは評価が高いらしいほかにも10個位tensorflow api にある
        self.step = cost, tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
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
                #最後は与えられた答えの数 positive negativeの2になる
                need_answer_count = self.data.answer_type_count
            else :
                #隠れ層がある場合はその層のニューロンの数？の答えをだすのね
                need_answer_count = layers[loop_count]
            if loop_count is 0 :
                # 最初は与えられた特徴なのね
                feature = self.feature
                feature_type_count = self.data.feature_type_count
            else :
                # 隠れ層がある場合は隠れ層が学習する特徴のデータになるんだ！
                feature = hidden_layer
                feature_type_count = layers[loop_count-1]
            # truncated_normal Tensorを正規分布かつ標準偏差0.0001の２倍までのランダムな値で初期化する
            weights = tf.Variable(tf.truncated_normal([feature_type_count, need_answer_count], stddev=0.0001, seed=seeds))
            # バイアス
            biases = tf.Variable(tf.ones([need_answer_count]))
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
            self.real_answer: self.data.traning.answers
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
                # コスト関数の表示
                print( i, "\t", loss_vec[-1])
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
        predictions = tf.argmax(self.model, 1)
        real_answers = tf.argmax(self.real_answer, 1)
        ones_real_answers = tf.ones_like(real_answers)
        zeros_real_answers = tf.zeros_like(real_answers)
        ones_predictions = tf.ones_like(predictions)
        zeros_predictions = tf.zeros_like(predictions)
        # 答え01, 予想01
        zeroone_zeroone = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(real_answers, ones_real_answers),
                    tf.equal(predictions, ones_predictions)
                ),
                tf.float32
            )
        )
        # 答え01, 予想10
        zeroone_onezero = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(real_answers, ones_real_answers),
                    tf.equal(predictions, zeros_predictions)
                ),
                tf.float32
            )
        )
        # 答え10, 予想01
        onezero_zeroone = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(real_answers, zeros_real_answers),
                    tf.equal(predictions, ones_predictions)
                ),
                tf.float32
            )
        )
        # 答え10, 予想10
        onezero_onezero = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(real_answers, zeros_real_answers),
                    tf.equal(predictions, zeros_predictions)
                ),
                tf.float32
            )
        )
        one_one, one_zero, zero_one, zero_zero = self.session.run(
            [onezero_onezero, onezero_zeroone, zeroone_onezero, zeroone_zeroone],
            feed_dict_t
        )
        print("Up_Accuracy\t=", one_one/(one_one+one_zero))
        print("Down_Accuracy\t=", zero_zero/(zero_one+zero_zero))
        print("Accuracy\t=", (one_one+zero_zero)/(one_one+one_zero+zero_one+zero_zero))

    def value(self, ANSWER):
        # test_list
        feed_dict_t = {
            self.feature: self.data.test.features,
            self.real_answer: self.data.test.answers
        }
        predictions = tf.argmax(self.model, 1)
        answer = self.session.run(predictions, feed_dict_t)
        data = np.array(feed_dict_t[self.feature][ANSWER])
        value = [0]
        for i in range(1, len(answer)):
            if answer[i-1] == 0: temp_value = data[i]
            else: temp_value = -data[i]
            value.append(temp_value+value[-1])
        plt.plot(value, 'k-', label='Asset volatility')
        plt.title('Asset volatility')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend(loc='lower right')
        plt.show()