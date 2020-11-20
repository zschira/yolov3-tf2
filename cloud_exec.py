import tensorflow_cloud as tfc
tfc.run(docker_image_bucket_name="zach_schira_bucket",
        entry_point="train.py",
)
