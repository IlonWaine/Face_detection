from logging import config, getLogger


LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,

    "formatters": {
        "simple": {
            "format": "%(levelname)s: %(message)s",
        },
        "verbose": {
            "format": "%(asctime)s [%(name)s] %(levelname)-8s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },

    "filters": {
        "only_app": {"name": "myapp"},
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "verbose",
            "filters": ["only_app"],
            "filename": "app.log",
            "maxBytes": 10_485_760,
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },

    "loggers": {
        "myapp": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },

    "root": {
        "level": "WARNING",
        "handlers": ["console"],
    },
}

config.dictConfig(LOGGING)

logger = getLogger("myapp")
