import ray


@ray.remote(num_gpus=8)
class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def get_counter(self):
        return self.value


# Create an actor from this class.
counter = Counter.remote()

# Call the actor.
obj_ref = counter.increment.remote()
print(ray.get(obj_ref))
