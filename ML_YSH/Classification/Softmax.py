import tensorflow as tf
import numpy as np

class Softmax:
    # 멤버변수
    # Variables
    # Data number
    x_num = None;
    y_num = None;
    w_num = None;
    data_num = training_num = testing_num = None;
    # For Data
    X_training = None
    Y_training = None
    X_testing = None
    Y_testing = None
    # For hypothesis
    W = None
    X = None
    Y = None
    hypothesis = None
    cost = None
    # Session
    sess = None

    # 생성자함수. 초기화.
    def __init__(self):
        self.X = tf.placeholder("float", [None, None], name="x_ph")  # x1, x2 and 1 (for bias)
        self.Y = tf.placeholder("float", [None, None], name="y_ph")  # A, B, C => 3 classes

        self.sess = tf.Session()

    # Set data
    # txt 파일명과 class 개수를 받으면, 데이터를 입력받는다.
    # gildong.set_data('file.txt', 3)와 같이 사용
    def set_data(self, txt_file_name, y_num):
        # load txt
        xy = np.loadtxt(txt_file_name, unpack=True, dtype='float32')
        self.data_num = len(xy[0, :])

        # Training set
        self.training_num = int(self.data_num * (4/5)) # 전체 data의 80%만 train

        self.X_training = np.transpose(xy[0:y_num, 0:self.training_num])
        self.Y_training = np.transpose(xy[y_num: , 0:self.training_num])

        print (self.X_training)
        print (self.Y_training)

        # Testing set
        self.X_testing = np.transpose(xy[0:y_num, self.training_num:]) # training data가 아닌 data
        self.Y_testing = np.transpose(xy[y_num: , self.training_num:])

        self.x_num = self.w_num = len(self.X_training[0, ])
        self.y_num = y_num

        print (self.w_num, self.x_num, self.y_num, self.data_num, self.training_num)
        print (tf.shape(self.X_training))

        # Multi variable 판별
        if len(self.X_training) == 2: self.IsMulti = False
        else: self.IsMulti = True

        print ("IsMulti : ", self.IsMulti)

    # Learn
    # Linear Regression의 학습과정 (w, b 찾기)
    # cost값이 finish_point 이하면 종료
    # gildong.learn(0.001)와 같이 사용
    def learn(self, finish_point):

        self.W = tf.Variable(tf.zeros([self.training_num, self.training_num]))

        # Our hypothesis
        self.hypothesis = tf.nn.softmax(tf.matmul(self.W, self.X))

        # cost function
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(self.hypothesis), reduction_indices=1))

        # Minimize
        a = tf.Variable(0.001)  # Learning rate
        optimizer = tf.train.GradientDescentOptimizer(a).minimize(self.cost)

        # initialize the variables
        init = tf.initialize_all_variables()

        # Launch the graph.
        self.sess.run(init)

        # Fit the line
        print("step, cost, W")

        for step in range(2001):
            self.sess.run(optimizer, feed_dict={self.X: self.X_training, self.Y: self.Y_training})
            if step % 200 == 0:
                print(step, self.sess.run(self.cost, feed_dict={self.X: self.X_training, self.Y: self.Y_training}), self.sess.run(self.W))

        # step = 0
        # while True:
        #     self.sess.run(train, feed_dict={self.X: self.X_training, self.Y: self.Y_training})
        #     step += 1
        #     if step % 20 == 0:
        #         print(step, self.sess.run(self.cost, feed_dict={self.X: self.X_training, self.Y: self.Y_training}), self.sess.run(self.W))
        #     if self.sess.run(self.cost, feed_dict={self.X: self.X_training, self.Y: self.Y_training}) < finish_point : # cost값이 일정 이하로 내려가면 함수 종료
        #         print(step, self.sess.run(self.cost, feed_dict={self.X: self.X_training, self.Y: self.Y_training}),
        #               self.sess.run(self.W))
        #         break

    # test 1
    # testing set을 이용한 학습 결과 test
    # gildong.test()와 같이 사용
    def test(self):
        prediction = self.sess.run(self.hypothesis, feed_dict={self.X: self.X_testing}) > 0.5
        label = self.Y_testing > 0.5

        print ("testing set을 통한 예측 : ", prediction)
        print ("실제 값 : ", label)

        if prediction.all() == label.all():
            print ("학습 success")
        else:
            print ("학습 fail")

    # 실제 수행 함수
    # logistic_classification() 사용 후 what_is_it() 사용
    def logistic_classification(self, txt_file_name):
        print("set_data")
        self.set_data(txt_file_name)
        print("learn")
        self.learn(0.1)
        print("test")
        self.test()

    # prediction
    # 학습결과를 토대로 예측
    # 매개변수로 x 배열(혹은 x)을 받음
    # gildong.what_is_it([3,4])와 같이 사용
    def what_is_it(self, input_data):
        # data input
        x_data = np.ones((self.x_num, 1))

        for i in range(0, self.x_num - 1):
            x_data[i+1, 0] = input_data[i]

        print ("input_data : ", input_data)
        print ("x_data : ", x_data)

        # prediction output
        print ("prediction : ",  self.sess.run(self.hypothesis, feed_dict={self.X:x_data}) > 0.5)

# main
gildong = Softmax()
# gildong.logistic_classification("train_LC.txt")
# gildong.what_is_it([3,4])
gildong.set_data("train_soft.txt", 3)
gildong.learn(0.01)