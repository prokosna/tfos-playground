import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflowonspark import TFNode
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split


def main_fun(args, ctx):
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    X = preprocessing.scale(X)
    # Y = to_categorical(Y, num_classes=3)

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)
    print(train_X.shape, test_X.shape, train_Y.shape, test_Y.shape)

    model = Sequential()
    model.add(Dense(12, input_shape=(4,), activation='relu'))
    model.add(Dense(3, input_shape=(12,), activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    model.summary()

    estimator = tf.keras.estimator.model_to_estimator(model, model_dir=args.model_dir)
#     model.fit(train_X, train_Y, nb_epoch=50, batch_size=1, verbose=1)

#     loss, accuracy = model.evaluate(test_X, test_Y, verbose=0)
#     print("Accuracy = {:.2f}".format(accuracy))
    tf_feed = TFNode.DataFeed(ctx.mgr)
    def rdd_generator():
        while not tf_feed.should_stop():
            batch = tf_feed.next_batch(1)
            if len(batch) > 0:
                record = batch[0]
                features = np.array(record[0]).astype(numpy.array)
                label = np.array(record[1]).astype(numpy.float32)
                yield (features, label)

    def train_input_fn():
        ds = tf.data.Dataset.from_generator(rdd_generator,
                                           (tf.array, tf.float32),
                                           (tf.TensorShape([4]), tf.TensorShape([3])))
        ds = ds.batch(args.batch_size)
        return ds

#     train_input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={"dense_input": train_X},
#         y=train_Y,
#         batch_size=1,
#         num_epochs=None,
#         shuffle=True
#     )
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"dense_input": test_X},
        y=test_Y,
        num_epochs=args.epochs,
        shuffle=False
    )

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=args.steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"dense_input": test_X[:1]},
        y=test_Y[:1],
        batch_size=1,
        shuffle=False
    )

#     preds = estimator.predict(input_fn=test_input_fn)
#     for pred in preds:
#         print(pred)


from pyspark.ml.linalg import Vectors

if __name__ == '__main__':
    from tensorflowonspark import TFCluster
    executors = sc._conf.get("spark.executor.instances")
    num_executors = int(executors) if executors is not None else 2
    num_ps = 1

    args = edict({
        "cluster_size": num_executors,
        "num_ps": num_ps,
        "tensorboard": False,
        "model_dir": "/spark/data",
        "epochs": 1,
        "steps": 2000
    })

    # iris RDD
    iris = datasets.load_iris()
    df = spark.createDataFrame([(int(target), Vectors.dense(data)) for target, data in zip(iris.target, iris.data)], ['label', 'features'])
    rdd = df.select('features', 'label').rdd.map(tuple)

    cluster = TFCluster.run(sc, main_fun, args, args.cluster_size, args.num_ps, args.tensorboard, TFCluster.InputMode.SPARK, master_node='master')
    cluster.train(rdd, args.epochs)
    cluster.shutdown()
