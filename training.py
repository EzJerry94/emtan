import tensorflow as tf
import time


class Train():

    def __init__(self, train_data_provider, batch_size, epochs, num_classes,
                 learning_rate, predictions, train_sample_num, save_path, scope):
        self.train_data_provider = train_data_provider
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.train_sample_num = train_sample_num
        self.predictions = predictions
        self.save_path = save_path
        self.scope = scope
        self.ckpt_path = './ckpt/multi/multi_model.ckpt'

    def start_training(self):
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(3)  # set random seed for initialization

            self.train_data_provider.get_batch()

            iterator = self.train_data_provider.dataset.make_initializable_iterator()
            frames, labels= iterator.get_next()

            labels = tf.one_hot(labels, depth=3, axis=-1)
            labels = tf.reshape(labels, (self.batch_size, self.num_classes))
            frames = tf.reshape(frames, (self.batch_size, -1, 640))

            train_prediction = self.predictions(frames, self.scope)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=train_prediction, labels=labels)
            cross_entropy_mean = tf.reduce_mean(loss, name='cross_entropy')
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy_mean)

            saver = tf.train.Saver()

        with tf.Session(graph=g) as sess:
            train_num_batches = int(self.train_sample_num / self.batch_size)
            #sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.ckpt_path)
            print("Model restored.")

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
            save_path = saver.save(sess, self.save_path)
            print("Model saved in path: %s" % save_path)