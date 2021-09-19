def display_tf_ds(tf_dataset, max_count):
    i = 0
    for item in tf_dataset.as_numpy_iterator():
        if i >= max_count:
            break
        print(item)
        i += 1
