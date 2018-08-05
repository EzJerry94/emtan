import tensorflow as tf
import time


class MultiTrain():

    def __init__(self, train_data_provider, batch_size, epochs, num_classes,
                 learning_rate, predictions):
        self.train_data_provider = train_data_provider
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.train_sample_num = 31155
        self.predictions = predictions

    def start_training(self):
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(3)  # set random seed for initialization

            self.train_data_provider.get_batch()

            iterator = self.train_data_provider.dataset.make_initializable_iterator()
            frames, arousals, valences, dominances= iterator.get_next()

            arousals = tf.one_hot(arousals, depth=3, axis=-1)
            arousals = tf.reshape(arousals, (self.batch_size, self.num_classes))
            valences = tf.one_hot(valences, depth=3, axis=-1)
            valences = tf.reshape(valences, (self.batch_size, self.num_classes))
            dominances = tf.one_hot(dominances, depth=3, axis=-1)
            dominances = tf.reshape(dominances, (self.batch_size, self.num_classes))
            frames = tf.reshape(frames, (self.batch_size, -1, 640))

            train_prediction = self.predictions(frames)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=train_prediction, labels=labels)
            cross_entropy_mean = tf.reduce_mean(loss, name='cross_entropy')
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy_mean)

            saver = tf.train.Saver()

        with tf.Session(graph=g) as sess:
            train_num_batches = int(self.train_sample_num / self.batch_size)
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.epochs):
                print('\n Start Training for epoch {}\n'.format(epoch + 1))
                sess.run(iterator.initializer)
                for batch in range(train_num_batches):
                    start_time = time.time()
                    _, loss_value = sess.run([optimizer, cross_entropy_mean])
                    time_step = time.time() - start_time
                    print("Epoch {}/{}: Batch {}/{}: loss = {:.4f} ({:.2f} sec/step)".format(
                        epoch + 1, self.epochs, batch + 1, train_num_batches, loss_value, time_step))

            print('\n Training Completed \n')
            save_path = saver.save(sess, './ckpt/single/arousal/model.ckpt')
            print("Model saved in path: %s" % save_path)