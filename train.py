import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    TreeNet, YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset


def main():
    model = TreeNet(True)
    model.summary()
    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks

    train_dataset = dataset.load_tree_tfrecord_dataset("gs://zach_schira_bucket/data.tfrecord")

    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(16)
    train_dataset = train_dataset.map(lambda x, y: (
        x,
        dataset.transform_targets(y, anchors, anchor_masks, 416)))

    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)


    # Configure the model for transfer learning

    optimizer = tf.keras.optimizers.Adam(lr=10e-3)
    loss = [YoloLoss(anchors[mask], classes=2)
            for mask in anchor_masks]

    model.compile(optimizer=optimizer, loss=loss,
                  run_eagerly=False)

    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=3, verbose=1),
        TensorBoard(log_dir='gs://zach_schira_bucket/logs')
    ]

    history = model.fit(train_dataset,
                        epochs=5000,
                        callbacks=callbacks
                        )
    model.save("gs://zach_schira_bucket/tree_model")


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
