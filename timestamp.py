import datetime
import pytz

START_TIMESTAMP = datetime.datetime.now(
    tz=pytz.timezone("Europe/Amsterdam")
).strftime('%d-%m-%Y--%H-%M')
