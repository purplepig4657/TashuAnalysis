class RentDataCN:
    RENT_STATION = 'rent_station'
    RENT_DATE = 'rent_date'
    RETURN_STATION = 'return_station'
    RETURN_DATE = 'return_date'
    DISTANCE = 'distance'
    USER_CATEGORY = 'user_category'
    RENT_LATITUDE = 'rent_latitude'
    RENT_LONGITUDE = 'rent_longitude'
    RETURN_LATITUDE = 'return_latitude'
    RETURN_LONGITUDE = 'return_longitude'
    RENT_COUNT = 'rent_count'


class LocationDataCN:
    ID = 'id'
    VERSION = 'version'
    STATION_NAME = 'station_name'
    GU = 'gu'
    DONG = 'dong'
    LATITUDE = 'latitude'
    LONGITUDE = 'longitude'


class WeatherDataCN:
    ID = 'id'
    NAME = 'name'
    DATE = 'date'
    TEMPERATURE = 'temperature'
    TEMPERATURE_AVG = 'temperature_avg'
    PRECIPITATION = 'precipitation'
    WIND_SPEED = 'wind_speed'
    HUMIDITY = 'humidity'
    SUNSHINE_DURATION = 'sunshine_duration'
    WEATHER_NUMBER = 'weather_number'
    GROUND_TEMPERATURE = 'ground_temperature'
    RAINFALL = 'rainfall'


class TimeDataCN:
    YEAR = 'year'
    MONTH = 'month'
    DAY = 'day'
    HOUR = 'hour'
    WEEKDAY = 'weekday'
    TIME_CATEGORY = 'time_category'


class ClusterDataCN:
    INDEX = 'index'
    STATION = 'station'
    CLUSTER = 'cluster'


class TimeDataValue:
    MORNING_PEAK = 'morning_peak'
    MORNING_NON_PEAK = 'morning_non_peak'
    AFTERNOON_PEAK = 'afternoon_peak'
    AFTERNOON_NON_PEAK = 'afternoon_non_peak'
    NIGHTTIME = 'nighttime'


class WeatherDataValue:
    RAIN = 'rain'
    NON_RAIN = 'non_rain'
    CLOUDY = 'cloudy'
    NON_CLOUDY = 'non_cloudy'
