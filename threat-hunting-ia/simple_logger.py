"""
    simple_logger.py

    This module contains a simple implementation of a logger that will make the
    CLI more intuitive.

    Author: Angel Casanova
    2023
"""
import logging
import matplotlib


class CustomFormatter(logging.Formatter):
    """
    Custom Formatter for logging which defines the shapes and the colors of the default logger.

    Attributes:
        blue_dark (str): ANSI escape sequence for dark blue color.
        blue (str): ANSI escape sequence for blue color.
        green (str): ANSI escape sequence for green color.
        lime (str): ANSI escape sequence for lime color.
        orange (str): ANSI escape sequence for orange color.
        yellow (str): ANSI escape sequence for yellow color.
        red (str): ANSI escape sequence for red color.
        purple (str): ANSI escape sequence for purple color.
        grey (str): ANSI escape sequence for grey color.
        reset (str): ANSI escape sequence to reset colors to default.
        format_hour (str): Format string for the log record's timestamp.
        format_level (str): Format string for the log record's level.
        format_funct (str): Format string for the log record's function name and line number.
        format_file (str): Format string for the log record's filename and line number.
        format_msg (str): Format string for the log record's message.
        FORMATS (dict[int, (str, str)]): Dictionary that associates a log level with primary and secondary colors.
    """
    # Colours for the log

    blue_dark: str = "\x1b[38;2;102;102;255m"
    blue: str = "\x1b[38;2;5;161;247m"

    green: str = "\x1b[38;2;24;240;24m"
    lime: str = "\x1b[38;2;102;255;102m"

    orange: str = "\x1b[38;2;255;153;51m"
    yellow: str = "\x1b[38;2;255;255;25m"

    red: str = "\x1b[38;2;255;0;0m"
    purple: str = "\x1b[38;2;255;0;255m"

    grey: str = "\x1b[38;2;192;192;192m"
    reset: str = "\x1b[0m"

    # Formats for the output
    format_hour: str = "%(asctime)s.%(msecs)03d "
    format_level: str = "%(levelname)s"
    format_funct: str = "(%(funcName)s:%(lineno)d) "
    format_file: str = "(%(filename)s:%(lineno)d) "
    format_msg: str = "%(message)s"

    # Specify the colours to each level of log
    FORMATS: dict[int, (str, str)] = {
        logging.DEBUG: (blue, blue_dark),
        logging.INFO: (green, lime),
        logging.WARNING: (orange, yellow),
        logging.ERROR: (red, purple),
        logging.CRITICAL: (red, purple)
    }

    def format(self, record: logging.LogRecord) -> str:
        """ Overrides the default behaviour of the logging.Formatter class in order to apply the
        custom shapes and colors to the defined chanel.
        Args:
            record (str): Logger record that is going to be parsed.
        Returns:
            _.
        """
        principal, secondary = self.FORMATS.get(record.levelno, (self.reset, self.reset))

        log_fmt: str | None = self.__apply_colors(principal, secondary)
        formatter: logging.Formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

    def __apply_colors(self, principal, secondary) -> str:
        return self.grey + self.format_hour + \
            secondary + "[" + principal + self.format_level + secondary + "] " + \
            self.reset + self.format_file + \
            principal + self.format_msg + \
            self.reset

    @staticmethod
    def setup_logging(app_name: str) -> None:
        """ Static method that can be called to set up the whole logger. After calling this function, you can get
         your custom logger just calling: logging.getLogger('app_name')
        Args:
            app_name (str): Name that is going to be used for discriminating the new custom logger.
        Returns:
            _.
        """
        # Creates a logger for the application.
        logger = logging.getLogger(app_name)
        logger.setLevel(logging.DEBUG)

        # Creates a logger handle for logging into stderr
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)

        # Avoid Debug messages about fonts when printing with matplotlib
        matplotlib.pyplot.set_loglevel(level='warning')
        logging.getLogger('matplotlib._get_ticker_locator_formatter').setLevel(logging.ERROR)
