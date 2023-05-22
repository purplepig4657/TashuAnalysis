from ydata_profiling import ProfileReport


class DataProfiler:
    def __init__(self, data: dict):
        self.__data = data

    def __generate_profile_instance(self):
        profile_instances = dict()
        for name, data in self.__data.items():
            profile_instances[name] = ProfileReport(data)
        return profile_instances

    # noinspection PyMethodMayBeStatic
    def __profile_instance_to_file(self, profile_instances: dict):
        for name, profile_instance in profile_instances.items():
            profile_instance.to_file(f"{name}.html")

    def profiling(self):
        profile_instances = self.__generate_profile_instance()
        self.__profile_instance_to_file(profile_instances)
